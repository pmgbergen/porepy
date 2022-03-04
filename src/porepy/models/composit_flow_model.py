""" 
Contains a general composit flow class. Phases and components can be added during the set-up.
Does not involve chemical reactions.

Large parts of this code are attributed to EK and his prototype of the reactive multiphase model.
VL refactored the model for usage with the composite submodule.
"""

from this import d
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

        # TODO find more modular solution
        # aperture per grid in gridbucket
        self._apertures = [1.]

        ## Representation of variables

        # primary differential variables are global pressure, global enthalpy
        # and component overall fractions
        self.pressure_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["pressure"]
        self.energy_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["enthalpy"]
        # NOTE: ambiguity energy-enthalpy, keeping option for different energy variable in future
        self.component_overall_fraction_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["component_overall_fraction"]

        # primary algebraic variables are component molar fractions per phase,
        # saturations (phase volumetric fractions) and phase molar fractions
        self.component_molar_fraction_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["component_fraction_in_phase"]
        self.saturation_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["saturation"]
        self.phase_molar_fraction_variable: str = pp.composite.COMPUTATIONAL_VARIABLES["phase_molar_fraction"]        

        ## Parameter keywords
        self.flow_parameter_key: str = "flow"
        self.upwind_parameter_key: str = "upwind"
        self.mass_parameter_key: str = "mass"

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
            - resolves model composition in terms of phases and components
                - stores information about anticipated phase change and initial conditions
            - sets model parameters
                - boundary conditions
                - source terms
                - porosity, permeability, aperture (constant for now)
            - assigns primary pressure and enthalpy variable
                - set initial conditions for them
            - sets the model equations using :module:`porepy.ad`
                - discretizes the equations
        """

        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.gb, self.params["file_name"], folder_name=self.params["folder_name"]
        )

        self._resolve_composition()
        
        self._set_parameters()

        # Assign variables. This will also set up DOF- and EquationManager,
        # and define Ad versions of the variables not related to the composition
        self._assign_variables()

        # NOTE if the initial conditions are not in equilibrium, it needs to be iterated prior to simulation
        self._initial_condition()

        # Set and discretize equations
        self._assign_equations()

        self._discretize()

#------------------------------------------------------------------------------
### SIMULATION related methods
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

#------------------------------------------------------------------------------
### SET-UP
#------------------------------------------------------------------------------

    def _resolve_composition(self) -> None:
        """
        Analyzes the associated :class:`~porepy.composite.computational_domain.ComputationalDomain`
        and obtains information about the composition, i.e. about phases and substances.
        
        Information about substances which are anticipated in multiple phases is stored.

        Computes initial overall molar fractions per component
        (see :method:`~porepy.composite.substance.Substance.overall_molar_fraction`)
        """
        pass

    #### collective set-up method
    def _set_parameters(self) -> None:
        """Set default parameters needed in the simulations.

        Many of the functions here may change in the future, partly to allow for more
        general descriptions of fluid and rock properties. Also, overriding some of
        the constitutive laws may lead to different parameters being needed.

        """
        for g, d in self.gb:

            bc, bc_vals = self._BC_convective_flux(g)

            mass_sources = self._mass_sources(g)

            # specific volume and aperture are related to other physics. This has to stay like this for now.
            specific_volume = self._specific_volume(g)

            material_subdomain = g(pp.composite.UnitSolid(self.cd))
            transmissability = pp.SecondOrderTensor(
                specific_volume * material_subdomain.base_permeability()
            )

            # No gravity
            gravity = np.zeros((self.gb.dim_max(), g.num_cells))
            # with gravity TODO add mass density
            # gravity = np.vstack([np.zeros(self.gb.num_cells),
            #                      np.zeros(self.gb.num_cells),
            #                      -np.ones(self.gb.num_cells)])

            pp.initialize_data(
                g,
                d,
                self.flow_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "source": source_vals,  # TODO: specify as MASS source term and split it according substances
                    "second_order_tensor": transmissability,
                    "vector_source": gravity.ravel("F"),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )

            # Mass weight parameter. Same for all phases
            mass_weight = material_subdomain.base_porosity() * specific_volume
            pp.initialize_data(
                g, d, self.mass_parameter_key, {"mass_weight": mass_weight}
            )
            
            # NOTE VL: below should be done per component, not phase, since we have an equation per component.

            # NOTE EK: Seen from the upstream discretization, the Darcy velocity is a
            # parameter, although it is a derived quantity from the flow discretization
            # point of view. We will set a value for this in the initialization, to
            # increase the chances the user remembers to set compatible flux and
            # pressure.

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
            d[pp.STATE][self.energy_variable][:] = 1  
            d[pp.STATE][pp.ITERATE][self.energy_variable][:] = 1

            # Careful here: Use the same variable to store the Darcy flux for all phases.
            # This is okay in this case, but do not do so for other variables.
            darcy = np.zeros(g.num_faces)
            for j in range(self.num_fluid_phases):
                d[pp.PARAMETERS][f"{self.upwind_parameter_key}_{j}"][
                    "darcy_flux"
                ] = darcy

    ### Boundary Conditions

    def _BC_convective_flux(self, g: pp.Grid) -> Tuple[pp.BoundaryCondition, np.ndarray]:
        """ NOTE: this assumes for now the simplest case, namely a single rectangular grid.
        
        Convective flux due to the pressure potential.

        Phys. Dimensions:
            - Dirichlet conditions: [Pa] = [N / m^2] = [kg / m^1 / s^2]
            - Neumann conditions: [Pa / A] = [kg / m^(dim) / s^2]  ([A] = [m^(1 OR 2)] depending on dimension dim)
              (physical dimension given without the scalar part of the total convective flux expression)


        :param g: grid representing subdomain on which BCs are imposed
        :type g: :class:`~porepy.grids.grid.Grid`
        
        :return: Returns the :class:`~porepy.params.bc.BoundaryCondition` object and respective values.
        :rtype: Tuple[porepy.BoundaryCondition, numpy.ndarray]
        """
        # change this value for the constant D-BC to change
        outflow_pressure = 1. 

        all_bf, east_bf, west_bf, north_bf, south_bf, *_ = self._domain_boundary_sides(g)

        # Dirichlet on west side, BoundaryCondition object automatically assumes Neumann for rest
        bc = pp.BoundaryCondition(g, west_bf, "dir")

        # Zero-Neumann BC on whole domain
        vals = np.zeros(g.num_faces)
        # Constant-Dirichlet BC on westside
        vals[west_bf] = outflow_pressure

        return (bc, vals)

    def _BC_conductive_flux(self, g: pp.Grid) -> Tuple[pp.BoundaryCondition, np.ndarray]:
        """ NOTE: this assumes for now the simplest case, namely a single rectangular grid.
        
        (Thermal) conductive flux due to the temperature potential

        Phys. Dimensions:
            - Dirichlet conditions: [K]
            - Neumann conditions: [J / s / m^2] = [kg / s^-3] ([J]=[kg m^2 / s^-2])


        :param g: grid representing subdomain on which BCs are imposed
        :type g: :class:`~porepy.grids.grid.Grid`
        
        :return: Returns the :class:`~porepy.params.bc.BoundaryCondition` object and respective values.
        :rtype: Tuple[porepy.BoundaryCondition, numpy.ndarray]
        """
        # change this value for the constant D-BC to change
        bottom_temperature = 373.15  # in Kelvin, this is 100°C 

        all_bf, east_bf, west_bf, north_bf, south_bf, *_ = self._domain_boundary_sides(g)

    def _bc_type_transport(self, g: pp.Grid, j: int) -> pp.BoundaryCondition:
        """Set type of boundary condition for transport of phase j"""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "neu")

    def _bc_values_transport(self, g: pp.Grid, j: int) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(g.num_faces)

    #### Source terms

    def _mass_sources(self, g: pp.Grid) -> Dict[str, np.ndarray]:
        """ Single-cell source term in center of first grid part
        |-----|-----|-----|
        |  .  |     |     |
        |-----|-----|-----|

        Phys. Dimensions: [mol / m^3 / s]

        :return: source values per cell and per component. Keys: names of component, Values: source terms
        :rtype: Dict[str, :class:`~numpy.ndarray`]
        """
        # change this value to alter source magnitude
        source = 1.

        # find and set single-cell source
        vals = np.zeros(g.num_cells)
        source_cell = g.closest_cell(np.ndarray([0.5, 0.5]))
        vals[source_cell] = source

        return {"SimpleWater": vals}

    def _enthalpy_source(self, g: pp.Grid) -> np.ndarray:
        """ Single-cell source term in center of first grid part

        NOTE: For the basic simulation, only water is pumped into the system

        |-----|-----|-----|
        |  .  |     |     |
        |-----|-----|-----|

        Phys. Dimensions: [J / mol / s] = [kg m^2 / mol / s^3]

        :return: source values per cell and per component.
        :rtype: :class:`~numpy.ndarray`
        """
        # Values for computing the enthalpy source term
        water_source_pressure = 3.
        water_source_temperature = 293.15  # 20°C water source

    #### Variable set-up

    def _assign_variables(self) -> None:
        """Define variables used to describe the system.

        These will include both primary and secondary variables, however, to be
        compatible to the terminology in PorePy, they will all be denoted as primary
        in certain settings.

        """
        # This function works in three steps:
        # 1) All variables are defined, with the right set of DOFS.
        # 2) Set Dof- and EquationManagers.
        # 3) Define AD representations of all the variables. This is done by
        #    calling a separate function.

        for g, d in self.gb:
            # Naming scheme for component and phase indices:
            # Component index is always i, phase index always j.
            primary_variables = {self.pressure_variable: {"cells": 1},
                                 self.energy_variable: {"cells": 1}
                                 }

            # Total molar fraction of each component
            primary_variables.update(
                {
                    f"{self.component_variable}_{i}": {"cells": 1}
                    for i in range(self.num_components)
                }
            )

            # Phase mole fractions. Only in fluid phases
            primary_variables.update(
                {
                    f"{self.phase_mole_fraction_variable}_{i}": {"cells": 1}
                    for i in range(self.num_fluid_phases)
                }
            )
            # Saturations. Only in fluid phases
            primary_variables.update(
                {
                    f"{self.saturation_variable}_{i}": {"cells": 1}
                    for i in range(self.num_fluid_phases)
                }
            )

            # Component phase molar fractions
            # Note systematic naming convention: i is always component, j is phase.
            for j in range(self.num_phases):
                for i in range(self.num_components):
                    if self._component_present_in_phase[i, j]:
                        primary_variables.update(
                            {f"{self.component_phase_variable}_{i}_{j}": {"cells": 1}}
                        )
            # The wording is a bit confusing here, these will not be taken as
            d[pp.PRIMARY_VARIABLES] = primary_variables

        for e, d in self.gb.edges():
            raise NotImplementedError("Have only considered non-fractured domains.")

        # All variables defined, we can set up Dof and Equation managers
        self.dof_manager = pp.DofManager(self.gb)
        self._eq_manager = pp.ad.EquationManager(self.gb, self.dof_manager)

        # The manager set, we can finally do the Ad formulation of variables
        self._assign_ad_variables()

    def _assign_ad_variables(self) -> None:
        """Make lists of AD-variables, indexed by component and/or phase number.
        The idea is to enable easy access to the Ad variables without having to
        construct these from the equation manager every time we need them.
        """
        eqm = self._eq_manager

        # type annotation removed since it also annotated in the class definition
        # Pylance throws warning otherwise
        self._ad.pressure = eqm.merge_variables(
            [(g, self.pressure_variable) for g in self._grids]
        )

        self._ad.enthalpy = eqm.merge_variables(
            [(g, self.energy_variable) for g in self._grids]
        )

        self._ad.component = []

        for i in range(self.num_components):
            name = f"{self.component_variable}_{i}"
            var = eqm.merge_variables([(g, name) for g in self._grids])
            self._ad.component.append(var)

        # Represent component phases as an numpy array instead of a list, so that we
        # can access items by array[i, j], rather the more cumbersome array[i][j]
        self._ad.component_phase = {}
        for i in range(self.num_components):
            # Make inner list
            for j in range(self.num_phases):
                if self._component_present_in_phase[i, j]:
                    name = f"{self.component_phase_variable}_{i}_{j}"
                    var = eqm.merge_variables([(g, name) for g in self._grids])
                    self._ad.component_phase[(i, j)] = var

        self._ad.saturation = []
        self._ad.phase_mole_fraction = []
        for j in range(self.num_fluid_phases):
            # Define saturation variables for each phase
            sat_var = eqm.merge_variables(
                [(g, f"{self.saturation_variable}_{j}") for g in self._grids]
            )
            self._ad.saturation.append(sat_var)

            # Define Molar fraction variables, one for each phase
            mf_var = eqm.merge_variables(
                [(g, f"{self.phase_mole_fraction_variable}_{j}") for g in self._grids]
            )
            self._ad.phase_mole_fraction.append(mf_var)

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
### MATHEMATICAL MODEL equation methods
#------------------------------------------------------------------------------

    def _assign_equations(self) -> None:
        """Method to set all equations."""

        # balance equations
        self._set_transport_equations()
        self._set_enthalpy_equation()            

        # Equilibrium equations
        self._phase_equilibrium_equations()
        self._chemical_equilibrium_equations()

        # Equations for pure bookkeeping, relations between variables etc.
        self._overall_molar_fraction_sum_equations()
        self._component_phase_sum_equations()
        self._phase_mole_fraction_sum_equation()

        self._saturation_definition_equation()

        # Now that all equations are set, we define sets of primary and secondary
        # equations, and similar with variables. These will be used to represent
        # the systems to be solved globally (transport equations) and locally
        # (equilibrium equations).
        eq_manager = self._eq_manager

        # What to do in the case of a single component is not clear to EK at the time
        # of writing. Question is, do we still eliminate (the only) one transport equation?
        # Maybe the answer is a trivial yes, but raise an error at this stage, just to
        # be sure.
        assert len(self._ad.component) > 1

        # FIXME: Mortar variables are needed here
        assert len(self._edges) == 0

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
            for name in list(eq_manager.equations.keys())
            if name[:12] != "Mass_balance"
        ]

        self._secondary_equation_manager = eq_manager.subsystem_equation_manager(
            secondary_equation_names, secondary_variables
        )

        # Also store the name of the primary variables, we will need this to construct
        # the global linear system later on.
        # FIXME: Should we also store secondary equation names, for symmetry reasons?
        self._primary_equation_names = list(
            set(eq_manager.equations.keys()).difference(secondary_equation_names)
        )

    #### Methods to set transport equation

    def _set_transport_equations(self) -> None:
        """Set transport equations"""

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

    #### Methods to set energy equation

    def _set_enthalpy_equation(self) -> None:
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

    #### Equilibrium and closing equations

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

    def _kinetic_reaction_rate(self, i: int) -> float:
        """Get the kinetic rate for a given reaction."""
        raise NotImplementedError("This is not covered")

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

    def _porosity(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous porosity"""
        if g.dim < self.gb.dim_max():
            # Unit porosity in fractures. Scaling with aperture (dimension reduction)
            # should be handled by including a specific volume.
            scaling = 1
        else:
            scaling = 0.2

        return np.zeros(g.num_cells) * scaling

    def _viscosity(self, g: pp.Grid) -> np.ndarray:
        """Unitary viscosity.

        Units: kg / m / s = Pa s
        """
        return np.ones(g.num_cells)

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
