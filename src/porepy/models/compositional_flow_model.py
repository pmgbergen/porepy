"""
Contains a general composit flow class. Phases and components can be added during the set-up.
Does not involve chemical reactions.

Large parts of this code are attributed to EK and his prototype of the
reactive multiphase model.
VL refactored the model for usage with the composite submodule.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp


class CompositionalFlowModel(pp.models.abstract_model.AbstractModel):
    """Non-isothermal flow consisting of water in liquid and vapor phase
    and salt in liquid phase.

    The phase equilibria equations for water are given using k-values. The composition class is
    instantiated in the constructor using a sub-routine which can be inherited and overwritten.

    Parameters:
        params: general model parameters including

            - 'use_ad' : Bool  -  indicates whether :module:`porepy.ad` is used or not
            - 'file_name' : str  -  name of file for exporting simulation results
            - 'folder_name' : str  -  absolute path to directory for saving simulation results

        monolithic_solver: flag for solving the monolithic solving strategy.
    """

    def __init__(self, params: dict, monolithic_solver: bool = True) -> None:
        super().__init__(params)

        ### PUBLIC
        ### MODEL TUNING
        self.params: dict = params
        # Pa
        self.initial_pressure = 1013200
        # Kelvin
        self.initial_temperature = 323.15
        # %
        self.initial_salt_concentration = 0.02
        # kg to mol
        self.injected_moles_water = 10.0 / pp.composite.IAPWS95_H2O.molar_mass()
        # kJ / mol , specific heat capacity from Wikipedia
        self.injected_water_enthalpy = (
            0.075327 * self.initial_temperature
            + self.initial_pressure / (998.21 / pp.composite.IAPWS95_H2O.molar_mass())
        )
        # D-BC temperature Kelvin for conductive flux
        self.boundary_temperature = 383.15  # 110 Celsius
        # D-BC on outflow boundary
        self.boundary_pressure = 1.0
        # base porosity for grid
        self.porosity = 1.0
        # base permeability for grid
        self.permeability = 1.0
        # solver strategy. If monolithic, the model will take care of flash calculations
        # if not, it will solve only the primary system
        self.monolithic = monolithic_solver

        # time step size
        self.dt: float = 1.0
        # residual tolerance for the balance equations
        self.tolerance_balance_equations = 1e-8
        # create default grid bucket for this model
        self.mdg: pp.MixedDimensionalGrid
        self.box: dict = dict()
        self.create_grid()

        # contains information about the primary system
        self.flow_subsystem: dict[str, list] = dict()

        # Parameter keywords
        self.flow_parameter_key: str = "flow"
        self.flow_upwind_parameter_key: str = "upwind_%s" % (self.flow_parameter_key)
        self.upwind_parameter_key: str = "upwind"
        self.mass_parameter_key: str = "mass"
        self.energy_parameter_key: str = "energy"
        self.conduction_upwind_parameter_key: str = "upwind_%s" % (
            self.energy_parameter_key
        )

        # references to discretization operators
        # they will be set during `prepare_simulation`
        self.darcy_flux: pp.ad.MpfaAd
        self.darcy_upwind: pp.ad.UpwindAd
        self.darcy_upwind_bc: pp.ad.BoundaryCondition
        self.conductive_flux: pp.ad.MpfaAd
        self.conductive_upwind: pp.ad.UpwindAd
        self.conductive_upwind_bc: pp.ad.BoundaryCondition

        ### GEOTHERMAL MODEL SET UP
        self.composition: pp.composite.Composition
        self.brine: pp.composite.Phase
        self.vapor: pp.composite.Phase
        self.water: pp.composite.FluidComponent
        self.salt: pp.composite.SolidComponent

        self.set_composition()

        # Use the managers from the composition so that the Schur complement can be made
        self.eqm: pp.ad.EquationManager = self.composition.eq_manager
        self.dofm: pp.DofManager = self.composition.dof_manager

        ### PRIVATE
        # list of primary and secondary equations
        self._prim_equ: list[str] = list()
        self._sec_equ: list[str] = list()
        # list of primary and secondary variables
        self._prim_vars: list[pp.ad.Variable] = list()
        self._sec_vars: list[pp.ad.Variable] = list()
        # projections from global dofs to primary and secondary variables
        # will be set during `prepare simulations`
        self._proj_prim: sps.spmatrix
        self._proj_sec: sps.spmatrix
        # system variables and names
        # will also be set during preparations
        self._system_vars: tuple[list, list]
        # list of grids as ordered in GridBucket
        self._grids = [g for g in self.mdg.subdomains()]
        # list of edges as ordered in GridBucket
        self._edges = [e for e in self.mdg.interfaces()]

        ## model-specific input.
        self._source_quantity = {
            "H2O": self.injected_moles_water,  # mol in 1 m^3 (1 mol of liquid water is approx 1.8xe-5 m^3)
            "NaCl": 0.0,  # only water is pumped into the system
        }
        self._source_enthalpy = {
            "H2O": self.injected_water_enthalpy,  # in kJ
            "NaCl": 0.0,  # no new salt enters the system
        }

    def create_grid(self) -> None:
        """Assigns a cartesian grid as computational domain.
        Overwrites the instance variables 'gb'.
        """
        refinement = 1
        phys_dims = [1, 1]
        n_cells = [i * refinement for i in phys_dims]
        g = pp.CartGrid(n_cells, phys_dims)
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        g.compute_geometry()
        self.mdg = pp.meshing._assemble_in_bucket([[g]])

    def set_composition(self) -> None:
        """Define the composition for which the simulation should be run and performs
        the initial (p-T) equilibrium calculations for the initial state given in p-T and feed.

        Use this method to inherit and override the composition, while keeping the (generic)
        rest of the model.
        """
        self.water = pp.composite.IAPWS95_H2O(self.mdg)
        self.salt = pp.composite.NaCl(self.mdg)
        self.composition = pp.composite.Composition(self.mdg)
        self.brine = pp.composite.IncompressibleFluid("brine", self.mdg)
        self.vapor = pp.composite.IdealGas("vapor", self.mdg)
        self.brine.add_component([self.water, self.salt])
        self.vapor.add_component(self.water)

        self.composition.add_component(self.water)
        self.composition.add_component(self.salt)
        self.composition.add_phase(self.brine)
        self.composition.add_phase(self.vapor)

        k_value = 0.9
        name = "k_value_" + self.water.name
        equilibrium = (
            self.water.fraction_in_phase(self.vapor.name)
            - k_value * self.water.fraction_in_phase(self.brine.name)
        )

        self.composition.add_equilibrium_equation(self.water, equilibrium, name)

        self.composition.initialize()

        self.composition.set_feed_composition(
            [1. - self.initial_salt_concentration, self.initial_salt_concentration]
        )
        self.composition.set_state(self.initial_pressure, self.initial_temperature)

        self.composition.isothermal_flash(copy_to_state=True, initial_guess="feed")
        self.composition.evaluate_specific_enthalpy()
        self.composition.evaluate_saturations()

    def prepare_simulation(self) -> None:
        """Preparing essential simulation configurations.

        Method needs to be called after the composition has been set and prior to applying any
        solver.
        """

        # Exporter initialization for saving results
        self.exporter = pp.Exporter(
            self.mdg, self.params["file_name"], folder_name=self.params["folder_name"]
        )

        # Define primary and secondary variables/system which are secondary and primary in
        # the composition subsystem
        primary_vars = list()
        primary_var_names = list()
        self.flow_subsystem.update(
            {
                "primary_equations": [],
                "primary_vars": primary_vars,
                "primary_var_names": primary_var_names,
            }
        )

        secondary_vars = self.composition.ph_subsystem["primary_vars"]
        secondary_var_names = self.composition.ph_subsystem[
            "primary_var_names"
        ]
        self.flow_subsystem.update({"secondary_vars": secondary_vars})
        self.flow_subsystem.update({"secondary_var_names": secondary_var_names})

        sec_eq = [eq for eq in self.composition.ph_subsystem["equations"]]
        self.flow_subsystem.update({"secondary_equations": sec_eq})

        self._set_up()
        self._set_overall_fraction_sum_equation()
        self._set_mass_balance_equations()
        self._set_energy_balance_equation()

        # NOTE this can be optimized. Some component need to be discretized only once.
        self.eqm.discretize(self.mdg)
        # prepare datastructures for the solver
        self._prim_vars = self.eqm._variables_as_list(
            self.flow_subsystem["primary_vars"]
        )
        self._sec_vars = self.eqm._variables_as_list(
            self.flow_subsystem["secondary_vars"]
        )
        self._prim_equ = self.flow_subsystem["primary_equations"]
        self._sec_equ = self.flow_subsystem["secondary_equations"]
        self._proj_prim = self.eqm._column_projection(self._prim_vars)
        self._proj_sec = self.eqm._column_projection(self._sec_vars)
        
        # the system vars depend on whether we solve it monolithically or not
        if self.monolithic:
            vars = (
                self.flow_subsystem["primary_vars"]
                + self.flow_subsystem["secondary_vars"]
            )
            names = (
                self.flow_subsystem["primary_var_names"]
                + self.flow_subsystem["secondary_var_names"]
            )
            self._system_vars = (vars, names)
        else:
            self._system_vars = (
                self.flow_subsystem["primary_vars"],
                self.flow_subsystem["primary_var_names"],
            )

    # ------------------------------------------------------------------------------
    ### SIMULATION related methods and implementation of abstract methods
    # ------------------------------------------------------------------------------

    def before_newton_loop(self) -> None:
        """Resets the iteration counter and convergence status."""
        self.convergence_status = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Re-discretizes the Upwind operators and the fluxes."""
        # Darcy flux upwinding
        # compute the flux
        kw = self.flow_parameter_key
        kw_store = self.flow_upwind_parameter_key
        variable = self.composition._pressure_var
        pp.fvutils.compute_darcy_flux(self.mdg, kw, kw_store, p_name=variable)
        # re-discretize the upwinding of the Darcy flux
        self.darcy_upwind.upwind.discretize(self.mdg)
        self.darcy_upwind.bound_transport_dir.discretize(self.mdg)
        self.darcy_upwind.bound_transport_neu.discretize(self.mdg)

        # compute the heat flux
        kw = self.energy_parameter_key
        kw_store = self.conduction_upwind_parameter_key
        variable = self.composition._temperature_var
        pp.fvutils.compute_darcy_flux(self.mdg, kw, kw_store, p_name=variable)
        # re-discretize the upwinding of the conductive flux
        self.conductive_upwind.upwind.discretize(self.mdg)
        self.conductive_upwind.bound_transport_dir.discretize(self.mdg)
        self.conductive_upwind.bound_transport_neu.discretize(self.mdg)

    def after_newton_iteration(
        self, solution_vector: np.ndarray, iteration: int
    ) -> None:
        """
        Distributes solution of iteration additively to the iterate state of the variables.
        Increases the iteration counter.

        :param solution_vector: solution to global linear system of current iteration
        :type solution_vector: numpy.ndarray
        """
        self._nonlinear_iteration += 1
        vars, var_names = self._system_vars
        solution_vector = self.composition._prolongation_matrix(vars) * solution_vector

        self.dofm.distribute_variable(
            values=solution_vector,
            variables=var_names,
            additive=True,
            to_iterate=True,
        )

        if not self._monolithic:
            print(".. starting equilibrium calculations at iteration %i" % (iteration))
            equilibrated = self.composition.isenthalpic_flash(
                self._max_iter_flash,
                self._tol_flash,
                copy_to_state=False,
                trust_region=self._use_TRU,
                eliminate_unity=self._elimination,
            )
            if not equilibrated:
                raise RuntimeError(
                    "Isenthalpic flash failed in iteration %i." % (iteration)
                )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Distributes the values from the iterate state to the the state
        (for next time step).
        Exports the results.
        """
        _, var_names = self._system_vars
        self.dofm.distribute_variable(solution, variables=var_names)

        if not self._monolithic:
            # TODO check if Schur complement expansion should be used sa initial guess
            # or new state
            print(
                ".. starting equilibrium calculations after iteration %i converged"
                % (iteration_counter)
            )
            equilibrated = self.composition.isenthalpic_flash(
                self._max_iter_flash,
                self._tol_flash,
                copy_to_state=True,
                trust_region=self._use_TRU,
                eliminate_unity=self._elimination,
            )
            if not equilibrated:
                raise RuntimeError(
                    "Isenthalpic flash failed after iteration %i converged."
                    % (iteration_counter)
                )

        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Reset iterate state to previous state."""
        X = self.dofm.assemble_variable()
        self.dofm.distribute_variable(X, to_iterate=True)

    def after_simulation(self) -> None:
        """Does nothing currently."""
        pass

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """APerforms a Newton step for the whole system in a monolithic way, by constructing
        a Schur complement using the equilibrium equations and non-primary variables.

        :return: If converged, returns the solution. Until then, returns the update.
        :rtype: numpy.ndarray
        """

        if self.monolithic:
            A, b = self.eqm.assemble()
        else:
            # non-linear Schur complement elimination of secondary variables
            A_pp, b_p = self.eqm.assemble_subsystem(self._prim_equ, self._prim_vars)
            A_sp, _ = self.eqm.assemble_subsystem(self._sec_equ, self._prim_vars)
            A_ps, _ = self.eqm.assemble_subsystem(self._prim_equ, self._sec_vars)
            A_ss, b_s = self.eqm.assemble_subsystem(self._sec_equ, self._sec_vars)
            if self.composition._last_inverted is not None:
                inv_A_ss = self.composition._last_inverted
            else:
                inv_A_ss = np.linalg.inv(A_ss.A)

            A = A_pp - A_ps * inv_A_ss * A_sp
            A = sps.csr_matrix(A)
            b = b_p  # - A_ps * inv_A_ss * b_s

        res_norm = np.linalg.norm(b)
        if res_norm < tol:
            self.convergence_status = True
            _, var_names = self._system_vars
            x = self.dofm.assemble_variable(variables=var_names, from_iterate=True)
            return x

        # TODO direct solver only nice to small systems
        dx = spla.spsolve(A, b)

        return dx

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

    def _print(self, where=""):
        print("-------- %s" % (str(where)))
        print("Iterate")
        print(self.dofm.assemble_variable(from_iterate=True))
        print("State")
        print(self.dofm.assemble_variable())

    # ------------------------------------------------------------------------------
    ### SET-UP
    # ------------------------------------------------------------------------------

    #### collective set-up method
    def _set_up(self) -> None:
        """Set model components including
            - source terms,
            - boundary values,
            - permeability tensor

        A modularization of the solid skeleton properties is still missing.
        """

        for g, d in self.mdg.subdomains(return_data=True):

            source = self._unitary_source(g)
            unit_tensor = pp.SecondOrderTensor(np.ones(g.num_cells))
            # TODO is this a must-have parameter?
            zero_vector_source = np.zeros((self.mdg.dim_max(), g.num_cells))

            ### Mass weight parameter. Same for all balance equations
            pp.initialize_data(
                g,
                d,
                self.mass_parameter_key,
                {"mass_weight": self.porosity * np.ones(g.num_cells)},
            )

            ### Mass parameters per substance

            for component in self.composition.components:
                pp.initialize_data(
                    g,
                    d,
                    "%s_%s" % (self.mass_parameter_key, component.name),
                    {
                        "source": np.copy(
                            source * self._source_quantity[component.name]
                        ),
                    },
                )

            ### Darcy flow parameters for assumed single pressure
            bc, bc_vals = self._bc_advective_flux(g)
            transmissibility = pp.SecondOrderTensor(
                self.permeability * np.ones(g.num_cells)
            )
            # parameters for MPFA discretization
            pp.initialize_data(
                g,
                d,
                self.flow_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "second_order_tensor": transmissibility,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                },
            )
            # parameters for upwinding
            pp.initialize_data(
                g,
                d,
                self.flow_upwind_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "darcy_flux": np.zeros(
                        g.num_faces
                    ),  # Upwinding expects an initial flux
                },
            )

            ### Energy parameters for global energy equation
            bc, bc_vals = self._bc_conductive_flux(g)
            # enthalpy sources due to substance mass source
            param_dict = dict()
            for component in self.composition.components:
                param_dict.update(
                    {
                        "source_%s"
                        % (component.name): np.copy(
                            source * self._source_enthalpy[component.name]
                        )
                    }
                )
            # other enthalpy sources e.g., hot skeleton
            param_dict.update({"source": np.copy(source) * 0.0})
            # MPFA parameters for conductive flux
            param_dict.update(
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "second_order_tensor": unit_tensor,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                }
            )
            pp.initialize_data(
                g,
                d,
                self.energy_parameter_key,
                param_dict,
            )
            # parameters for conductive upwinding
            pp.initialize_data(
                g,
                d,
                self.conduction_upwind_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "darcy_flux": np.zeros(g.num_faces),  # Upwinding expects
                },
            )

        # For now we consider only a single domain
        for e, data_edge in self.mdg.interfaces(return_data=True):
            raise NotImplementedError("Mixed dimensional case not yet available.")

        ### Instantiating discretization operators

        # darcy flux
        mpfa = pp.ad.MpfaAd(self.flow_parameter_key, self._grids)
        bc = pp.ad.BoundaryCondition(self.flow_parameter_key, self._grids)
        self.darcy_flux = mpfa.flux * self.composition.pressure + mpfa.bound_flux * bc

        # darcy upwind
        keyword = self.flow_upwind_parameter_key
        self.darcy_upwind = pp.ad.UpwindAd(keyword, self._grids)
        self.darcy_upwind_bc = pp.ad.BoundaryCondition(keyword, self._grids)

        # conductive flux
        mpfa = pp.ad.MpfaAd(self.energy_parameter_key, self._grids)
        bc = pp.ad.BoundaryCondition(self.energy_parameter_key, self._grids)
        self.conductive_flux = (
            mpfa.flux * self.composition.temperature + mpfa.bound_flux * bc
        )
        # conductive upwind
        keyword = self.conduction_upwind_parameter_key
        self.conductive_upwind = pp.ad.UpwindAd(keyword, self._grids)
        self.conductive_upwind_bc = pp.ad.BoundaryCondition(keyword, self._grids)

    ### Boundary Conditions

    def _bc_advective_flux(self, g: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for advective flux (Darcy). Override for modifications.

        Phys. Dimensions of ADVECTIVE MASS FLUX:
            - Dirichlet conditions: [Pa] = [N / m^2] = [kg / m^1 / s^2]
            - Neumann conditions: [mol / m^2 s] = [(mol / m^3) * (m^3 / m^2 s)]
                (molar density * Darcy flux)

        Phys. Dimensions of ADVECTIVE ENTHALPY FLUX:
            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [J m^3 / m^2 s] (density * specific enthalpy * Darcy flux)

        Phys. Dimensions of FICK's LAW OF DIFFUSION:
            - Dirichlet conditions: [-] (molar, fractional: constant concentration at boundary)
            - Neumann conditions: [mol / m^2 s]
              (same as advective flux)

        NOTE: Enthalpy flux D-BCs need some more thoughts.
        Isn't it unrealistic to assume the temperature or enthalpy of the outflowing fluid is
        known?
        That BC would influence our physical setting and it's actually our goal to find out
        how warm the water will be at the outflow.
        NOTE: BC has to be defined for all fluxes separately.
        """
        bc, vals = self._bc_unitary_flux(g, "east", "dir")
        return (bc, vals * self.boundary_pressure)

    def _bc_diff_disp_flux(self, g: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for diffusive-dispersive flux (Darcy). Override for modifications.

        Phys. Dimensions of FICK's LAW OF DIFFUSION:
            - Dirichlet conditions: [-] (molar, fractional: constant concentration at boundary)
            - Neumann conditions: [mol / m^2 s]
              (same as advective flux)
        """
        bc, vals = self._bc_unitary_flux(g, "east", "neu")
        return (bc, vals * self.boundary_pressure)

    def _bc_conductive_flux(
        self, g: pp.Grid
    ) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """Conductive BC for Fourier flux in energy equation. Override for modifications.

        Phys. Dimensions of CONDUCTIVE HEAT FLUX:
            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [J m^3 / m^2 s] (density * specific enthalpy * Darcy flux)
              (same as convective enthalpy flux)
        """
        bc, vals = self._bc_unitary_flux(g, "south", "dir")
        return (bc, vals * self.boundary_temperature)

    def _bc_unitary_flux(
        self, g: pp.Grid, side: str, bc_type: Optional[str] = "neu"
    ) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC objects for unitary flux on specified grid side.

        :param g: grid representing subdomain on which BCs are imposed
        :type g: :class:`~porepy.grids.grid.Grid`
        :param side: side with non-zero values ('west', 'north', 'east', 'south')
        :type side: str
        :param bc_type: (default='neu') defines the type of the `side` BC.
            Currently only Dirichlet and Neumann BC are supported
        :type bc_type: str

        :return: :class:`~porepy.params.bc.BoundaryCondition` instance and respective values
        :rtype: Tuple[porepy.BoundaryCondition, numpy.ndarray]
        """

        side = str(side)
        if side == "west":
            _, _, idx, *_ = self._domain_boundary_sides(g)
        elif side == "north":
            _, _, _, idx, *_ = self._domain_boundary_sides(g)
        elif side == "east":
            _, idx, *_ = self._domain_boundary_sides(g)
        elif side == "south":
            _, _, _, _, idx, *_ = self._domain_boundary_sides(g)
        else:
            raise ValueError(
                "Unknown grid side '%s' for unitary flux. Use 'west', 'north',..."
                % (side)
            )

        # non-zero on chosen side, the rest is zero-Neumann by default
        bc = pp.BoundaryCondition(g, idx, bc_type)

        # homogeneous BC
        vals = np.zeros(g.num_faces)
        # unitary on chosen side
        vals[idx] = 1.0

        return (bc, vals)

    #### Source terms

    def _unitary_source(self, g: pp.Grid) -> np.ndarray:
        """Unitary, single-cell source term in center of first grid part
        |-----|-----|-----|
        |  .  |     |     |
        |-----|-----|-----|

        Phys. Dimensions:
            - mass source:          [mol / m^3 / s]
            - enthalpy source:      [J / m^3 / s] = [kg m^2 / m^3 / s^3]

        :return: source values per cell.
        :rtype: :class:`~numpy.array`
        """
        # find and set single-cell source
        vals = np.zeros(g.num_cells)
        source_cell = g.closest_cell(np.array([0.5, 0.5]))
        vals[source_cell] = 1.0

        return vals

    # ------------------------------------------------------------------------------------------
    ### MODEL EQUATIONS: Mass and Energy Balance
    # ------------------------------------------------------------------------------------------

    def _set_overall_fraction_sum_equation(self) -> None:
        """Sets the overall substance fraction sum equation."""

        name = "overall_substance_fraction_sum"
        self.flow_subsystem["primary_equations"].append(name)

        fraction_sum = self.composition.overall_substance_fractions_unity()

        # index reduction of algebraic unitarity constraint
        # demand exponential decay of rate of change
        time_derivative = list()
        for substance in self.composition.unique_substances:
            time_derivative.append(
                substance.overall_fraction
                - substance.overall_fraction.previous_timestep()
            )
        decay = self.dt / (2 * np.pi)

        eq = sum(time_derivative) + self.dt * decay * fraction_sum

        self.eqm.equations.update({name: eq})

    def _set_mass_balance_equations(self) -> None:
        """Set mass balance equations per substance"""

        # creating operators, parameters and shorter namespaces
        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, self._grids)
        cp = self.composition
        div = pp.ad.Divergence(grids=self._grids, name="Divergence")
        upwind = self.darcy_upwind
        bc_upwind = self.darcy_upwind_bc

        for subst in self.composition.unique_substances:
            ### ACCUMULATION
            accumulation = mass.mass * (
                subst.overall_fraction * cp.composit_molar_density()
                - subst.overall_fraction.previous_timestep()
                * cp.composit_molar_density(prev_time=True)
            )

            # ADVECTION
            scalar_part = list()
            for phase in cp.phases_of_substance(subst):

                scalar_ = (
                    phase.molar_density(cp.pressure, cp.temperature)
                    * subst.fraction_in_phase(phase.name)
                    * self.rel_perm(phase.saturation)  # TODO change rel perm access
                    / phase.dynamic_viscosity(cp.pressure, cp.temperature)
                )
                scalar_part.append(scalar_)
            scalar_part = sum(scalar_part)

            advection = (
                self.darcy_flux * (upwind.upwind * scalar_part)
                - upwind.bound_transport_dir * self.darcy_flux * bc_upwind
                - upwind.bound_transport_neu * bc_upwind
            )

            ### SOURCE
            keyword = "%s_%s" % (self.mass_parameter_key, subst.name)
            source = pp.ad.ParameterArray(keyword, "source", grids=self._grids)

            ### MASS BALANCE PER COMPONENT
            # minus in advection already included
            equ_subst = accumulation + self.dt * (div * advection - source)
            equ_name = "mass_balance_%s" % (subst.name)
            self.eqm.equations.update({equ_name: equ_subst})
            self.flow_subsystem["primary_equations"].append(equ_name)

    def _set_energy_balance_equation(self) -> None:
        """Sets the global energy balance equation in terms of enthalpy."""

        # creating operators, parameters and shorter namespaces
        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, self._grids)
        cp = self.composition
        div = pp.ad.Divergence(grids=self._grids, name="Divergence")
        upwind_adv = self.darcy_upwind
        bc_upwind_adv = self.darcy_upwind_bc
        upwind_cond = self.conductive_upwind
        bc_upwind_cond = self.conductive_upwind_bc

        ### ACCUMULATION
        accumulation = mass.mass * (
            cp.composit_specific_molar_enthalpy * cp.composit_molar_density()
            - cp.composit_specific_molar_enthalpy.previous_timestep()
            * cp.composit_molar_density(prev_time=True)
        )

        ### ADVECTION
        scalar_part = list()
        for phase in cp:
            scalar_ = (
                phase.molar_density(cp.pressure, cp.temperature)
                * phase.specific_molar_enthalpy(cp.pressure, cp.temperature)
                * self.rel_perm(phase.saturation)  # TODO change rel perm access
                / phase.dynamic_viscosity(cp.pressure, cp.temperature)
            )
            scalar_part.append(scalar_)
        scalar_part = sum(scalar_part)

        advection = (
            self.darcy_flux * (upwind_adv.upwind * scalar_part)
            - upwind_adv.bound_transport_dir * self.darcy_flux * bc_upwind_adv
            - upwind_adv.bound_transport_neu * bc_upwind_adv
        )

        ### CONDUCTION
        scalar_part = list()
        for phase in cp:
            scalar_part.append(
                phase.saturation
                * phase.thermal_conductivity(cp.pressure, cp.temperature)
            )
        scalar_part = mass.mass * sum(scalar_part)

        conduction = (
            self.conductive_flux * (upwind_cond.upwind * scalar_part)
            - upwind_cond.bound_transport_dir * self.conductive_flux * bc_upwind_cond
            - upwind_cond.bound_transport_neu * bc_upwind_cond
        )

        ### SOURCE
        # rock enthalpy source
        source = pp.ad.ParameterArray(
            self.energy_parameter_key, "source", grids=self._grids
        )
        # enthalpy source due to mass source
        for subst in cp.unique_substances:
            kw = "source_%s" % (subst.name)
            source += pp.ad.ParameterArray(
                self.energy_parameter_key, kw, grids=self._grids
            )

        ### GLOBAL ENERGY BALANCE
        equ_energy = accumulation + self.dt * (div * (advection + conduction) - source)
        equ_name = "energy_balance"
        self.eqm.equations.update({equ_name: equ_energy})
        self.flow_subsystem["primary_equations"].append(equ_name)

    # ------------------------------------------------------------------------------------------
    ### CONSTITUTIVE LAWS
    # ------------------------------------------------------------------------------------------

    def rel_perm(self, saturation: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Helper function until data structure for heuristic laws is done."""
        return saturation * saturation

    # def _aperture(self, g: pp.Grid) -> np.ndarray:
    #     """
    #     Aperture is a characteristic thickness of a cell, with units [m].
    #     1 in matrix, thickness of fractures and "side length" of cross-sectional
    #     area/volume (or "specific volume") for intersections of dimension 1 and 0.
    #     See also specific_volume.
    #     """
    #     aperture = np.ones(g.num_cells)
    #     if g.dim < self.gb.dim_max():
    #         aperture *= 0.1
    #     return aperture

    # def _specific_volume(self, g: pp.Grid) -> np.ndarray:
    #     """
    #     The specific volume of a cell accounts for the dimension reduction and has
    #     dimensions [m^(Nd - d)].
    #     Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
    #     of aperture in dimension 1 and 0.
    #     """
    #     a = self._aperture(g)
    #     return np.power(a, self.gb.dim_max() - g.dim)
