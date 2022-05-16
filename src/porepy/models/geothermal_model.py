""" 
Contains a general composit flow class. Phases and components can be added during the set-up.
Does not involve chemical reactions.

Large parts of this code are attributed to EK and his prototype of the reactive multiphase model.
VL refactored the model for usage with the composite submodule.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import porepy as pp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from ..composite._composite_utils import ConvergenceError


class GeothermalModel(pp.models.abstract_model.AbstractModel):
    """ Non-isothermal and non-isobaric flow consisting of water in liquid in vapor phase
    and salt in liquid phase.

    The phase equilibria equations for water are given using fugacities (k-values).

    Public properties:
        - gb : :class:`~porepy.grids.grid_bucket.GridBucket`
          Grid object for simulation (3:1 cartesian grid by default)
        - composition : :class:`~porepy.composite.composition.Composition`
        - box: 'dict' containing 'xmax','xmin' as keys and the respective bounding box values.
          Hold also for 'y' and 'z'.
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

        ### PUBLIC
        # create default grid bucket for this model
        self.gb: pp.GridBucket
        self.box: Dict = dict()
        self.create_grid()

        # Parameter keywords
        self.flow_parameter_key: str = "flow"
        self.upwind_parameter_key: str = "upwind"
        self.mass_parameter_key: str = "mass"
        self.energy_parameter_key: str = "energy"

        ### GEOTHERMAL MODEL SET UP TODO
        self.composition = pp.composite.Composition(self.gb)

        # Use the managers from the composition so that the Schur complement can be made
        self.eqm: pp.ad.EquationManager = self.composition.eq_manager
        self.dofm: pp.DofManager = self.composition.dof_manager
        
        ### PRIVATE
        # list of primary equations
        self._prim_equ: List[str] = list()
        # list of primary variables
        self._prim_var: List["pp.ad.MergedVariable"] = list()
        # list of names of primary variables (same order as above)
        self._prim_var_names: List[str] = list()
        # list of grids as ordered in GridBucket
        self._grids = [g for g, _ in self.gb]
        # list of edges as ordered in GridBucket
        self._edges = [e for e, _ in self.gb.edges()]
        # maximal number of iterations for flash and equilibrium calculations
        self._max_iter = 100
        # residual tolerance for flash and equilibrium calculations
        self._iter_eps = 1e-10

        ## model-specific input. NOTE hardcoded values are up for modularization
        # mol in 1 cubic meter (1 mol of liquid water is approx 1.8xe-5 m^3)
        # 0 mol salt source
        self._source_quantity = [55555.5, 0.]
        # enthalpy of water from source. TODO get physical value here
        # zero enthalpy from zero salt source
        self._source_enthalpy = [10., 0.]
        # 110°C temperature for D-BC for conductive flux
        self._conductive_boundary_temperature = 383.15
        # flux for N-BC
        self._outflow_flux = 1.
        # base porosity for grid
        self._base_porosity = 1.
        # base permeability for grid
        self._base_permeability = 1.

    def create_grid(self) -> None:
        """ Assigns a cartesian grid as computational domain.
        Overwrites the instance variables 'gb'.
        """
        refinement = 4
        phys_dims = [3, 1]
        n_cells = [i * refinement for i in phys_dims]
        g = pp.CartGrid(n_cells, phys_dims)
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        g.compute_geometry()
        self.gb = pp.meshing._assemble_in_bucket([[g]])

    def prepare_simulation(self) -> None:
        """
        Method needs to be called prior to applying any solver,
        and after adding relevant phases and substances.

        It does the following points:
            - model set-up
                - boundary conditions
                - source terms
                - connects to model parameters (constant for now)
                    - porosity
                    - permeability
                    - aperture
                - computes initial equilibrium of composition
            - sets the model equations using :module:`porepy.ad`
                - discretizes the equations
        """

        # Exporter initialization for saving results
        self.exporter = pp.Exporter(
            self.gb, self.params["file_name"], folder_name=self.params["folder_name"]
        )
        
        self._set_up()
        self._initial_condition()
        self._assign_equations()

        # after everything has been set, discretize
        self.eqm.discretize(self.gb)

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
        """Starts equilibrium computations and consequently flash calculations.
            1. Equilibrium
            2. Saturation Flash
            3. Isenthalpic Flash

        Throws an error if one of them fails to converge.
        """

        converged = self.composition.compute_phase_equilibrium(self._max_iter, self._iter_eps)
        if not converged:
            raise ConvergenceError("Equilibrium Calculations did not converge.")

        converged = self.composition.saturation_flash(self._max_iter, self._iter_eps)
        if not converged:
            raise ConvergenceError("Saturation Flash did not converge.")

        converged = self.composition.isenthalpic_flash(self._max_iter, self._iter_eps)
        if not converged:
            raise ConvergenceError("Isenthalpic Flash did not converge.")

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

#------------------------------------------------------------------------------
### SET-UP
#------------------------------------------------------------------------------

    #### collective set-up method
    def _set_up(self) -> None:
        """Set model components including
            - source terms,
            - boundary values,
            - permeability tensor

        A modularization of the solid skeleton properties is still missing.
        """
        
        for g, d, mat_sd in self.cd:

            bc, bc_vals = self._BC_unitary_transport_flux(g)
            # unitary source
            source = self._unitary_source(g)

            # transmissibility coefficients for the MPFA
            transmissability = pp.SecondOrderTensor(
                self._base_permeability
            )

            # No gravity
            # vector_source = np.zeros((self.gb.dim_max(), g.num_cells))

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
                    # "vector_source": vector_source.ravel("F"),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )
             # Mass weight parameter. Same for all phases
            pp.initialize_data(
                g, d, self.mass_parameter_key, {"mass_weight": self._base_porosity
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

        # For now we consider only a single domain
        for e, data_edge in self.gb.edges():
            raise NotImplementedError("Mixed dimensional case not yet available.")

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

    def _bc_advective_flux(self):
        """Advective flux"""

    def _bc_unitary_flux(
        self, g: "pp.Grid", side: str, bc_type: Optional[str] = "neu"
    ) -> Tuple[pp.BoundaryCondition, np.ndarray]:
        """BC objects for unitary flux on specified grid side.

        :param g: grid representing subdomain on which BCs are imposed
        :type g: :class:`~porepy.grids.grid.Grid`
        :param side: side with non-zero values ('west', 'north', 'east', 'south')
        :type side: str
        :param bc_type: (default='neu') defines the type of the eastside BC.
            Currently only Dirichlet and Neumann BC are supported
        :type bc_type: str
        
        :return: :class:`~porepy.params.bc.BoundaryCondition` instance and respective values
        :rtype: Tuple[porepy.BoundaryCondition, numpy.ndarray]
        """

        side = str(side)
        if side == "west":
            _, _, idx, *_ = self._domain_boundary_sides(g)
        if side == "north":
            _, _, _, idx, *_ = self._domain_boundary_sides(g)
        if side == "east":
            _, idx, *_ = self._domain_boundary_sides(g)
        if side == "south":
            _, _, _, _, idx,*_ = self._domain_boundary_sides(g)
        else:
            raise ValueError(
                "Unknown grid side '%s' for unitary flux. Use 'west', 'north',..." % (side)
                )

        # non-zero on chosen side, the rest is zero-Neumann by default
        bc = pp.BoundaryCondition(g, idx, bc_type)

        # homogeneous BC
        vals = np.zeros(g.num_faces)
        # unitary on chosen side
        vals[idx] = 1.

        return (bc, vals)

    def _BC_unitary_transport_flux(self, g: pp.Grid, bc_type: Optional[str] = "neu"
    )-> Tuple[pp.BoundaryCondition, np.ndarray]:
        """        
        BC objects for unitary flux.
        Currently only the east side allows boundary a non-zero flux

        Phys. Dimensions of ADVECTIVE MASS FLUX:
            - Dirichlet conditions: [Pa] = [N / m^2] = [kg / m^1 / s^2]
            - Neumann conditions: [mol / m^2 s] = [(mol / m^3) * (m^3 / m^2 s)] 
              (molar density * Darcy flux)
        
        Phys. Dimensions of ADVECTIVE ENTHALPY FLUX:
            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [J m^3 / m^2 s] (density * specific enthalpy * Darcy flux)

        Phys. Dimensions of FICK's LAW OF DIFFUSION:
            - Dirichlet conditions: [-] (molar, fractional: constant substance concentration at boundary) 
            - Neumann conditions: [mol / m^2 s]  (same as regular mass flux)

        NOTE: Enthalpy flux BCs need some more thoughts.
        Isn't it unrealistic to assume the temperature or enthalpy of the outflowing fluid is
        known?
        That BC would influence our physical setting and it's actually our goal to find out
        how warm the water will be at the outflow.
        NOTE: BC has to be defined for all fluxes separately.


        :param g: grid representing subdomain on which BCs are imposed
        :type g: :class:`~porepy.grids.grid.Grid`
        :param bc_type: (default='neu') defines the type of the eastside BC.
            Currently only Dirichlet and Neumann BC are supported
        :type bc_type: str
        
        :return: :class:`~porepy.params.bc.BoundaryCondition` instance and respective values
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

    #------------------------------------------------------------------------------------------
    ### MODEL EQUATIONS: Transport and Energy Balance
    #------------------------------------------------------------------------------------------

    def _assign_equations(self) -> None:
        """Method to set transport equations per substance and 1 global energy equations."""

        # balance equations
        self._set_mass_balance_equations()
        self._set_energy_balance_equation()            

        # TODO store primary equations, vars and var names

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

    def _darcy_flux(self, pressure: "pp.ad.MergedVariable") -> "pp.ad.Operator":
        """MPFA discretization of darcy potential for given pressure variable.

        :param pressure: a pressure variable
        :type pressure: :class:`~porepy.ad.MergedVariable`

        :return: MPFA discretization including boundary conditions
        :rtype: :class:`~porepy.ad.Operator`
        """
        mpfa = pp.ad.MpfaAd(self.flow_parameter_key, self._grids)

        bc = pp.ad.ParameterArray(self.flow_parameter_key, "bc_values", grids=self._grids)

        darcy = mpfa.flux * pressure + mpfa.bound_flux * bc
        return darcy

    def _upstream(self, phase_ind: int) -> pp.ad.Operator:
        # Not sure we need this one, but it may be convenient if we want to override this
        # (say, for countercurrent flow).

        upwind = pp.ad.UpwindAd(f"{self.upwind_parameter_key}_{phase_ind}", self._grids)

        rp = self._rel_perm(phase_ind)

        return upwind.upwind * rp

    #------------------------------------------------------------------------------------------
    ### CONSTITUTIVE LAWS
    #------------------------------------------------------------------------------------------

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
