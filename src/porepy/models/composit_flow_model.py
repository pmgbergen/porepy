""" 
Contains a general composit flow class. Phases and components can be added during the set-up.
Does not involve chemical reactions.

Large parts of this code are attributed to EK and his prototype of the reactive multiphase model.
VL refactored the model for usage with the composite submodule.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Union, Optional

import porepy as pp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Shorthand typing
interface_type = Tuple[pp.Grid, pp.Grid]
grid_like_type = Union[pp.Grid, interface_type]

class CompositionalFLow(pp.models.abstract_model.AbstractModel):
    """ Non-isothermal and non-isobaric flow consisting of multiple phases and components.

    Represents the mathematical model of compositional flow with phase change in molar formulation.
    Physical phase change model given by k-value approach.

    Public properties:
        - gb : :class:`~porepy.grids.grid_bucket.GridBucket` Grid object for simulation (3:1 cartesian grid by default)
        - cd : :class:`~porepy.composite.computational_domain.ComputationalDomain` based on 'gb'
        - box: 'dict' containing 'xmax','xmin' as keys and the respective bounding box values. Hold also for 'y' and 'z'

    """

    def __init__(self, params: Dict) -> None:
        """ Base constructor for a standard grid.

        The following configurations can be passed:
            - 'use_ad' : Bool  -  indicates whether :module:`porepy.ad` is used or not
            - 'file_name' : str  -  name of file for exporting simulation results (without extensions)
            - 'folder_name' : str  -  absolute path to directory for saving simulation results
        
        :param params: contains information about above configurations
        :type params: dict        
        """
        super().__init__(params)

        # create default grid bucket for this model
        self.gb: pp.GridBucket
        self.box: Dict = dict()
        self.create_grid()

        # public properties
        self.cd = pp.composite.ComputationalDomain(self.gb)
        
        # list of grids as ordered in GridBucket
        self._grids = [g for g, _ in self.gb]
        # list of edges as ordered in GridBucket
        self._edges = [e for e, _ in self.gb.edges()]

        # variable holding all involved component instances
        self._components = list() 
        # variable holding all involved fluid phases
        self._fluid_phases = list()

        # model-specific input. below hardcoded values are up for modularization
        self._water_source_quantity = 55555.5  # mol in 1 cubic meter (1 mol of liquid water is approx 1.8xe-5 m^3)
        self._water_source_pressure = 3.
        self._water_source_temperature = 293.15  # 20°C water source
        self._conductive_boundary_temperature = 383.15  # 110°C southern boundary temperature (D-BC)
        self._outflow_flux = 1.  # N-BC eastern boundary

        ## Representation of variables

        # main primary variables are global pressure, global enthalpy
        # and component overall fractions
        self.pressure_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["pressure"]
        self.enthalpy_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["enthalpy"]
        self.component_overall_fraction_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["component_overall_fraction"]

        # other primary variables are component molar fractions per phase,
        # saturations (phase volumetric fractions) and phase molar fractions
        self.component_molar_fraction_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["component_fraction_in_phase"]
        self.saturation_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["saturation"]
        self.phase_molar_fraction_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["phase_molar_fraction"]        

        ## Parameter keywords
        self.flow_parameter_key: str = "flow"
        self.upwind_parameter_key: str = "upwind"
        self.mass_parameter_key: str = "mass"
        self.energy_parameter_key: str = "energy"

        ## global AD variables
        self.pressure = self.cd(self.pressure_variable)
        self.enthalpy = self.cd(self.enthalpy_variable)

        # update the DOfs since two new variables have been added
        self.cd.dof_manager.update_dofs()

    @property
    def num_components(self) -> int:
        """
        :return: number of components in model
        :rtype: int
        """
        return len(self._components)
    
    @property
    def num_fluid_phases(self) -> int:
        """
        :return: number of fluid phases in model
        :rtype: int
        """
        return len(self._fluid_phases)

    @property
    def num_solid_phases(self) -> int:
        """
        :return: number of (immobile) solid phases. Currently returns always zero (in hindsight on future extensions)
        :rtype: int
        """
        return 0

    def create_grid(self) -> None:
        """ Assigns a cartesian grid as computational domain.
        Overwrites the instance variables 'gb'.
        """
        refinement = 4
        phys_dims = [3, 1]
        n_cells = [i * refinement for i in phys_dims]
        g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        self.box: Dict = pp.geometry.bounding_box.from_points(
            np.array([[0, 0], phys_dims]).T
        )
        g.compute_geometry()
        self.gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])

    def prepare_simulation(self) -> None:
        """
        Method needs to be called prior to applying any solver,
        and after adding relevant phases and substances.

        It does the following points:
            - model set-up
                - initiates primary variables enthalpy and pressure
                - initiates EQ and DOF managers
                - boundary conditions
                - source terms
                - connects to model parameters (constant for now)
                    - porosity
                    - permeability
                    - aperture
            - sets the model equations using :module:`porepy.ad`
                - discretizes the equations
        """

        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.gb, self.params["file_name"], folder_name=self.params["folder_name"]
        )
        
        self._set_up()

        # Assign variables. This will also set up DOF- and EquationManager,
        # and define Ad versions of the variables not related to the composition
        self._assign_variables()

        # NOTE if the initial conditions are not in equilibrium, it needs to be iterated prior to simulation
        self._initial_condition()

        # Set and discretize equations
        self._assign_equations()

        self._discretize()

#------------------------------------------------------------------------------
### SIMULATION related methods and implementation of abstract methods
#------------------------------------------------------------------------------

    def before_newton_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        """
        self.convergence_status = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        Solve the non-linear problem formed by the secondary equations. Put this in a
        separate function, since this is surely something that should be streamlined
        to the characteristics of each problem.

        """
        self._solve_secondary_equations()

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE][pp.ITERATE] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        self._nonlinear_iteration += 1
        self.dof_manager.distribute_variable(
            values=solution_vector, additive=True, to_iterate=True
        )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        solution = self.dof_manager.assemble_variable(from_iterate=True)

        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._is_nonlinear_problem():
            raise ValueError("Newton iterations did not converge")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleaup etc."""
        pass

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """Assemble the linearized system, described by the current state of the model,
        solve and return the new solution vector.

        Parameters:
            tol (double): Target tolerance for the linear solver. May be used for
                inexact approaches.

        Returns:
            np.array: Solution vector.

        """
        """Use a direct solver for the linear system."""

        eq_manager = self._eq_manager

        # Inverter for the Schur complement system. We have an inverter for block
        # diagnoal systems ready but I never got around to actually using it (this
        # is not difficult to do).
        inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

        primary_equations = [
            self._eq_manager[name] for name in self._primary_equation_names
        ]
        primary_variables = self._primary_variables()

        # This forms the Jacobian matrix and residual for the primary variables,
        # where the secondary variables are first discretized according to their
        # current state, and the eliminated using a Schur complement technique.
        A_red, b_red = eq_manager.assemble_schur_complement_system(
            primary_equations=primary_equations,
            primary_variables=primary_variables,
            inverter=inverter,
        )

        # Direct solver for the global linear system. Again, this is simple but efficient
        # for sufficiently small problems.
        x = sps.linalg.spsolve(A_red, b_red)

        # Prolongation from the primary to the full set of variables
        prolongation = self._prolongation_matrix(primary_variables)

        x_full = prolongation * x

        return x_full

    def _prolongation_matrix(self, variables) -> sps.spmatrix:
        # Construct mappings from subsects of variables to the full set.
        nrows = self.dof_manager.num_dofs()
        rows = np.unique(
            np.hstack(
                # The use of private variables here indicates that something is wrong
                # with the data structures. Todo..
                [
                    self.dof_manager.grid_and_variable_to_dofs(s._g, s._name)
                    for var in variables
                    for s in var.sub_vars
                ]
            )
        )
        ncols = rows.size
        cols = np.arange(ncols)
        data = np.ones(ncols)

        return sps.coo_matrix((data, (rows, cols)), shape=(nrows, ncols)).tocsr()

    def _solve_secondary_equations(self):
        """Solve the cell-wise non-linear equilibrium problem of secondary variables
        and equations.

        This is a simplest possible approach, using Newton's method, with a primitive
        implementation to boot. This should be improved at some point, however, approaches
        tailored to the specific system at hand, and/or outsourcing to dedicated libraries
        are probably preferrable.
        """
        # Equation manager for the secondary equations
        sec_man = self._secondary_equation_manager

        # The non-linear system will be solved with Newton's method. However, to get the
        # assembly etc. correctly, the updates during iterations should be communicated
        # to the data storage in the GridBucket. With the PorePy data model, this is
        # most conveniently done by the DofManager's method to distribute variables.
        # This method again can be tailored to target specific grids and variables,
        # but for simplicity, we create a prolongation matrix to the full set of equations
        # and use the resulting vector.
        prolongation = self._prolongation_matrix(self._secondary_variables())

        max_iters = 100
        i = 0
        while i < max_iters:
            A, b = sec_man.assemble()
            if np.linalg.norm(b) < 1e-10:
                break
            x = spla.spsolve(A, b)
            full_x = prolongation * x
            self.dof_manager.distribute_variable(full_x, additive=True, to_iterate=True)

            i += 1
        if i == max_iters:
            raise ValueError("Newton for local systems failed to converge.")

    def _discretize(self) -> None:
        """Discretize all terms"""
        self._eq_manager.discretize(self.gb)

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

    def _primary_variables(self) -> List[pp.ad.MergedVariable]:
        """Get a list of the primary variables of the system on AD form.

        This will be the pressure and n-1 of the total molar fractions.

        """
        # The primary variables are the pressure and all but one of the total
        # molar fractions.
        # Represent primary variables by their AD format, since this is what is needed
        # to interact with the EquationManager.
        primary_variables = [self._ad.pressure] + self._ad.component[:-1]
        return primary_variables

    def _secondary_variables(self) -> List[pp.ad.MergedVariable]:
        """Get a list of secondary variables of the system on AD form.

        This will the final total molar fraction, phase molar fraction, component
        mole fractions, and saturations.
        """
        # The secondary variables are the final molar fraction, saturations, phase
        # mole fractions and component phases.
        secondary_variables = (
            [self._ad.component[-1]]
            + self._ad.saturation
            + self._ad.phase_mole_fraction
            + list(self._ad.component_phase.values())
        )
        return secondary_variables

#------------------------------------------------------------------------------
### SET-UP
#------------------------------------------------------------------------------

    #### collective set-up method
    def _set_up(self) -> None:
        """Set default parameters needed in the simulations.

        Many of the functions here may change in the future, partly to allow for more
        general descriptions of fluid and rock properties. Also, overriding some of
        the constitutive laws may lead to different parameters being needed.

        """
        for g, d, mat_sd in self.cd:

            bc, bc_vals = self._BC_unitary_transport_flux(g)

            # TODO get enthalpy of inflow substance at given pressure and temperature
            inflow_enthalpy = 1.

            mass_source = self._unitary_source(g) * self._water_source_quantity
            enthalpy_source = np.copy(mass_source) * inflow_enthalpy

            # specific volume and aperture are related to other physics.
            # This has to stay like this for now.
            # specific_volume = self._specific_volume(g)

            # transmissibility coefficients for the mpfa
            transmissability = pp.SecondOrderTensor(
                mat_sd.base_permeability() # * specific_volume
            )

            # No gravity
            gravity = np.zeros((self.gb.dim_max(), g.num_cells))
            # With gravity FIXME WIP
            # gravity = np.array([0.,0.,-9.98])
            # gravity_glob = list()
            # for phase in self.cd.Phases:
            #     gravity_glob.append(phase.mass_phase_density() * gravity.copy())
            
            # gravity_glob = np.vstack(gravity_glob).T.ravel("F")

            # TODO split flow parameters into parameters per component (different sources)
            pp.initialize_data(
                g,
                d,
                self.flow_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "source": mass_source,  
                    "second_order_tensor": transmissability,
                    "vector_source": gravity.ravel("F"),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )
             # Mass weight parameter. Same for all phases
            pp.initialize_data(
                g, d, self.mass_parameter_key, {"mass_weight": mat_sd.base_porosity() # * specific_volume
                }
            )

            # NOTE EK: Seen from the upstream discretization, the Darcy velocity is a
            # parameter, although it is a derived quantity from the flow discretization
            # point of view. We will set a value for this in the initialization, to
            # increase the chances the user remembers to set compatible flux and
            # pressure.

            # NOTE VL: below should be done per component, not phase, since we have an equation per component.

            for j in range(self.num_fluid_phases):
                bc = self._bc_type_transport(g, j)
                bc_values = self._bc_values_transport(g, j)
                pp.initialize_data(
                    g,
                    d,
                    f"{self.upwind_parameter_key}_{j}",
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                    },
                )

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            raise NotImplementedError("Only single grid for now")

    ### Initial Conditions

    def _initial_condition(self):
        """Set initial conditions: Homogeneous for all variables except the pressure.
        Also feed a zero Darcy flux to the upwind discretization.
        """
        # This will set homogeneous conditions for all variables
        super()._initial_condition()

        for g, d in self.gb:
            d[pp.STATE][self.pressure_variable][:] = 1
            d[pp.STATE][pp.ITERATE][self.pressure_variable][:] = 1

            # intrinsic energy in [joule/moles]
            d[pp.STATE][self.enthalpy_variable][:] = 1  
            d[pp.STATE][pp.ITERATE][self.enthalpy_variable][:] = 1

            # Careful here: Use the same variable to store the Darcy flux for all phases.
            # This is okay in this case, but do not do so for other variables.
            darcy = np.zeros(g.num_faces)
            for j in range(self.num_fluid_phases):
                d[pp.PARAMETERS][f"{self.upwind_parameter_key}_{j}"][
                    "darcy_flux"
                ] = darcy

    ### Boundary Conditions

    def _BC_unitary_transport_flux(self, g: pp.Grid, bc_type: Optional[str] = "neu"
    )-> Tuple[pp.BoundaryCondition, np.ndarray]:
        """        
        BC objects for unitary flux. Currently only the east side allows boundary a non-zero flux

        Phys. Dimensions of CONVECTIVE MASS FLUX:
            - Dirichlet conditions: [Pa] = [N / m^2] = [kg / m^1 / s^2]
            - Neumann conditions: [mol / m^2 s] = [(mol / m^3) * (m^3 / m^2 s)]  (molar density * Darcy flux)
        
        Phys. Dimensions of CONVECTIVE ENTHALPY FLUX:
            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [J m^3 / m^2 s] (density * specific enthalpy * Darcy flux)
        NOTE: Enthalpy flux BCs need some more thoughts. Isn't it unrealistic to assume the temperature or enthalpy of the outflowing fluid is known?
        That BC would influence our physical setting and it's actually our goal to find out how warm the water will be at the outflow.
        NOTE: DC-BC for enthalpy might be tricky if pressure is given as DC-BC for convective mass flux at the same time (h = h(T,p))

        Phys. Dimensions of FICK's LAW OF DIFFUSION:
            - Dirichlet conditions: [-] (molar, fractional: constant substance concentration at boundary) 
            - Neumann conditions: [mol / m^2 s]  (same as regular mass flux)
        NOTE: Does the type of BC have to be the same for convective and diffusive flux? Or is anything else nonphysical?


        :param g: grid representing subdomain on which BCs are imposed
        :type g: :class:`~porepy.grids.grid.Grid`
        :param bc_type: (default='neu') defines the type of the eastside BC. Currently only Dirichlet and Neumann BC are supported
        :type bc_type: str
        
        :return: Returns the :class:`~porepy.params.bc.BoundaryCondition` object and respective values.
        :rtype: Tuple[porepy.BoundaryCondition, numpy.ndarray]
        """
        _, east_bf, *_ = self._domain_boundary_sides(g)

        # heterogeneous on west side, BoundaryCondition object automatically assumes  Neumann for rest
        bc = pp.BoundaryCondition(g, east_bf, bc_type)

        # homogeneous Neumann BC
        vals = np.zeros(g.num_faces)
        # Constant, unitary D-BC on eastside
        vals[east_bf] = 1.

        return (bc, vals)

    def _BC_unitary_conductive_flux(self, g: pp.Grid, bc_type: Optional[str] = "neu"
    )-> Tuple[pp.BoundaryCondition, np.ndarray]:
        """
        BC object for unitary, conductive flux. Currently only non-zero BC at southside assumed.

        Phys. Dimensions of CONDUCTIVE HEAT FLUX:
            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [J m^3 / m^2 s] (density * specific enthalpy * Darcy flux) (same as convective flux)
        """
        _, _, _, _, south_bf,*_ = self._domain_boundary_sides(g)

        # heterogeneous on west side, BoundaryCondition object automatically assumes  Neumann for rest
        bc = pp.BoundaryCondition(g, south_bf, bc_type)

        # homogeneous Neumann BC
        vals = np.zeros(g.num_faces)
        # Constant, unitary D-BC on eastside
        vals[south_bf] = 1.

        return (bc, vals)

    #### Source terms

    def _unitary_source(self, g: pp.Grid) -> np.array:
        """ Unitary, single-cell source term in center of first grid part
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
        source_cell = g.closest_cell(np.ndarray([0.5, 0.5]))
        vals[source_cell] = 1.

        return vals

#------------------------------------------------------------------------------
### MATHEMATICAL MODEL equation methods
#------------------------------------------------------------------------------

    def _assign_equations(self) -> None:
        """Method to set all equations."""

        # balance equations
        self._set_mass_balance_equations()
        self._set_energy_balance_equation()            

        # Equilibrium equations
        self._phase_equilibrium_equations()

        # Closing equations (unitarity conditions and definitions)
        self._overall_molar_fraction_sum_equations()
        self._component_phase_sum_equations()
        self._phase_mole_fraction_sum_equation()
        self._saturation_definition_equation()

        # Now that all equations are set, we define sets of primary and secondary
        # equations, and similar with variables. These will be used to represent
        # the systems to be solved globally (transport equations) and locally
        # (equilibrium equations).
        # Create a separate EquationManager for the secondary variables and equations.
        # This set of secondary equations will still contain the primary variables,
        # but their derivatives will not be computed in the construction of the
        # Jacobian matrix (strictly speaking, derivatives will be computed, then dropped).
        # Thus, the secondary manager can be used to solve the local (to cells) systems
        # describing equilibrium.

        # Get the secondary variables of the system.
        secondary_variables = self._secondary_variables()

        # Ad hoc approach to get the names of the secondary equations. This is not beautiful.
        secondary_equation_names = [
            name
            for name in list(self.cd.eq_manager.equations.keys())
            if name[:12] != "Mass_balance"
        ]

        self._secondary_equation_manager = self.cd.eq_manager.subsystem_equation_manager(
            secondary_equation_names, secondary_variables
        )

        # Also store the name of the primary variables, we will need this to construct
        # the global linear system later on.
        # FIXME: Should we also store secondary equation names, for symmetry reasons?
        self._primary_equation_names = list(
            set(self.cd.eq_manager.equations.keys()).difference(secondary_equation_names)
        )

    #### Balance equations

    def _set_mass_balance_equations(self) -> None:
        """Set transport equations"""
        # TODO treat case of single component system specially!
        darcy = self._single_phase_darcy()

        component_flux = [0 for i in range(self.num_components)]

        for j in range(self.num_fluid_phases):
            rp = self._rel_perm(j)
            visc = self._phase_viscosity(j)

            upwind = pp.ad.UpwindAd(f"{self.upwind_parameter_key}_{j}", self._grids)

            rho_j = self._density(j)

            darcy_j = (upwind.upwind * rp / visc) * darcy

            for i in range(self.num_components):
                if self._component_present_in_phase[i, j]:
                    component_flux[i] += darcy_j * (
                        upwind.upwind * (rho_j * self._ad.component_phase[i, j])
                    )
        
        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, self._grids)

        dt = 1

        rho_tot = self._density()
        rho_tot_prev_time = self._density(prev_time=True)

        div = pp.ad.Divergence(self._grids, dim=1, name="Divergence")

        component_mass_balance: List[pp.ad.Operator()] = []

        g = self.gb.grids_of_dimension(self.gb.dim_max())[0]

        for i in range(self.num_components):
            # Boundary conditions
            bc = pp.ad.ParameterArray(  # Not sure about this part - should there also be a phase-wise boundary condition?
                param_keyword=upwind.keyword, array_keyword="bc_values", grids=[g]
            )
            # The advective flux is the sum of the internal (computed in flux_{i} above)
            # and the boundary condition
            # FIXME: We need to account for both Neumann and Dirichlet boundary conditions,
            # and likely do some filtering.
            adv_flux = component_flux[i] + upwind.bound_transport_neu * bc

            z_i = self._ad.component[i]
            # accumulation term
            accum = (
                mass.mass
                * (z_i / rho_tot - z_i.previous_timestep() / rho_tot_prev_time)
                / dt
            )

            # Append to set of conservation equations
            component_mass_balance.append(accum + div * adv_flux)

        for i, eq in enumerate(component_mass_balance):
            self._eq_manager.equations[f"Mass_balance_component{i}"] = eq

    def _set_energy_balance_equation(self) -> None:
        """ Sets the global enthalpy balance equation.
        """

        darcy = self._single_phase_darcy()

        # redundant 
        # rp = [self._rel_perm(j) for j in range(self.num_fluid_phases)]

        component_flux = [0 for i in range(self.num_components)]

        for j in range(self.num_fluid_phases):
            rp = self._rel_perm(j)

            upwind = pp.ad.UpwindAd(f"{self.upwind_parameter_key}_{j}", self._grids)

            rho_j = self._density(j)

            darcy_j = (upwind.upwind * rp) * darcy

            for i in range(self.num_components):
                if self._component_present_in_phase[i, j]:
                    component_flux[i] += darcy_j * (
                        upwind.upwind * (rho_j * self._ad.component_phase[i, j])
                    )
    def _single_phase_darcy(self) -> pp.ad.Operator:
        """Discretize single-phase Darcy's law using Mpfa.

        Override method, e.g., to use Tpfa instead.

        Returns
        -------
        darcy : TYPE
            DESCRIPTION.

        """
        mpfa = pp.ad.MpfaAd(self.flow_parameter_key, self._grids)

        bc = pp.ad.ParameterArray(self.flow_parameter_key, "bc_values", grids=self._grids)

        darcy = mpfa.flux * self._ad.pressure + mpfa.bound_flux * bc
        return darcy

    def _upstream(self, phase_ind: int) -> pp.ad.Operator:
        # Not sure we need this one, but it may be convenient if we want to override this
        # (say, for countercurrent flow).

        upwind = pp.ad.UpwindAd(f"{self.upwind_parameter_key}_{phase_ind}", self._grids)

        rp = self._rel_perm(phase_ind)

        return upwind.upwind * rp

    #### Equilibrium equations

    def _phase_equilibrium_equations(self) -> None:
        """Define equations for phase equilibrium and assign to the EquationManager.

        For the moment, no standard 'simplest' model is implemented - this may change
        in the future.
        """
        if self.num_fluid_phases > 1:
            raise NotImplementedError(
                "Fluid phase equilibrium must be calculated in subproblem"
            )

    def _overall_molar_fraction_sum_equations(self) -> None:
        """
        Set equation zeta_i = \sum_j chi_ij * xi_j
        """
        eq_manager = self._eq_manager

        for i in range(self.num_components):

            phase_sum_i = sum(
                [
                    self._ad.component_phase[i, j] * self._ad.phase_mole_fraction[j]
                    for j in range(self.num_fluid_phases)
                    if self._component_present_in_phase[i, j]
                ]
            )

            eq = self._ad.component[i] - phase_sum_i
            eq_manager.equations[f"Overall_comp_phase_comp_{i}"] = eq

    def _component_phase_sum_equations(self) -> None:
        """Force the component phases to sum to unity for all components.

        \sum_i x_i,0 = \sum_i x_ij, j=1,..
        """
        eq_manager = self._eq_manager

        def _comp_sum(j: int) -> pp.ad.Operator:
            return sum(
                [
                    self._ad.component_phase[i, j]
                    for i in range(self.num_components)
                    if self._component_present_in_phase[i, j]
                ]
            )

        sum_0 = _comp_sum(0)

        for j in range(1, self.num_fluid_phases):
            sum_j = _comp_sum(j)
            eq_manager.equations[f"Comp_phase_sum_{j}"] = sum_0 - sum_j

    def _phase_mole_fraction_sum_equation(self) -> None:
        """Force mole fractions to sum to unity

        sum_j v_j = 1

        """
        eq = sum(
            [self._ad.phase_mole_fraction[j] for j in range(self.num_fluid_phases)]
        )
        unity = pp.ad.Array(np.ones(self.gb.num_cells()))
        self._eq_manager.equations["Phase_mole_fraction_sum"] = eq - unity

    def _saturation_definition_equation(self) -> None:
        """Relation between saturations and phase mole fractions"""
        weighted_saturation = [
            self._ad.saturation[j] * self._density(j)
            for j in range(self.num_fluid_phases)
        ]

        for j in range(self.num_fluid_phases):
            eq = self._ad.phase_mole_fraction[j] - weighted_saturation[j] / sum(
                weighted_saturation
            )
            self._eq_manager.equations[f"saturation_definition_{j}"] = eq

#------------------------------------------------------------------------------
### CONSTITUTIVE LAWS
#------------------------------------------------------------------------------

    def _rel_perm(self, j: int) -> pp.ad.Operator:
        """Get the relative permeability for a given phase.

        The implemented function is a quadratic Brooks-Corey function. Override to
        use a different function.

        IMPLEMENTATION NOTE: The data structure for relative permeabilities may change
        substantially in the future. Specifically, hysteretic effects and heterogeneities
        may require separate classes for flow functions.

        Parameters:
            j (int): Index of the phase.

        Returns:
            pp.ad.Operator: Relative permeability of the given phase.

        """
        sat = self._ad.saturation[j]
        # Brooks-Corey function
        return pp.ad.Function(lambda x: x ** 2, "Rel. perm. liquid")(sat)

    def _density(
        self, j: Optional[int] = None, prev_time: Optional[bool] = False
    ) -> pp.ad.Operator:
        """Get the density of a specified phase, or a saturation-weighted sum over
        all phases.

        Optionally, the density can be evaluated at the previous time step.

        The implemented function is that of a slightly compressible fluid. Override
        to use different functions, including functions calculated from external
        packages.

        FIXME: Should this be a public function?

        Parameters:
            j (int, optional): Index of the target phase. If not provided, a saturation
                weighted mean density will be returned.
            prev_time (bool, optional): If True, the density is evaluated at the previous
             time step. Defaults to False.

        Returns:
            pp.ad.Operator: Ad representation of the density.

        """
        if j is None:
            average = sum(
                [
                    self._density(j, prev_time) * self._ad.saturation[j]
                    for j in range(self.num_fluid_phases)
                ]
            )
            return average

        # Set some semi-random values for densities here. These could be set in the
        # set_parameter method (will give more flexibility), or the density could be
        # provided as a separate function / class (perhaps better suited to accomodate
        # external libraries)
        base_pressure = 1
        base_density = [1000, 800]
        compressibility = [1e-6, 1e-5]

        var = self._ad.pressure.previous_timestep() if prev_time else self._ad.pressure
        return pp.ad.Function(
            lambda p: base_density[j] * (1 + compressibility[j] * (p - base_pressure)),
            f"Density_phase_{j}",
        )(var)

    def _phase_enthalpy(self, phase_ind:int) -> pp.ad.Operator:
        """ Returns the phase-related partial enthalpy.

        NOTE: the structure of this quantity will change in future.
        Will be given by a different class most likely.
        
        Parameters:
            phase_ind (int): Index of the phase.

        Returns:
            pp.ad.Operator: Relative permeability of the given phase.
        """

        if phase_ind == 0:
            phase_enthalpy = lambda x: x
        elif phase_ind == 1:
            phase_enthalpy = lambda x: x
        else:
            raise RuntimeError("Phase enthalpy index out of range.")

        return pp.ad.Function(phase_enthalpy, f"Phase_enthalpy_{phase_ind}")(self._ad.pressure)
    
    def _phase_viscosity(self, phase_ind:int) -> pp.ad.Array:
        """ Returns the phase viscosity.
        Returns currently only the unitary values for all phases.

        NOTE: the structure of this quantity will change in future.
        Will be given by a different class most likely.
        
        Parameters:
            j (int): Index of the phase.

        Returns:
            pp.ad.Array: Relative permeability of the given phase.
        """
        return pp.ad.Array(np.ones(self.gb.num_cells()))

    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self.gb.dim_max():
            aperture *= 0.1
        return aperture

    def _specific_volume(self, g: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in dimension 1 and 0.
        """
        a = self._aperture(g)
        return np.power(a, self.gb.dim_max() - g.dim)
