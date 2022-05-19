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


class CompositionalFlowModel(pp.models.abstract_model.AbstractModel):
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
        # time step size
        self.dt: float = 1.
        # maximal number of iterations for flash and equilibrium calculations
        self.max_iter_equilibrium: int = 100
        # residual tolerance for flash and equilibrium calculations
        self.tolerance_equilibrium = 1e-10
        # residual tolerance for the balance equations
        self.tolerance_balance_equations = 1e-10
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
        
        self.primary_subsystem: Dict[str, list] = {
            "equations": ["overall_component_fraction_sum"],
            "vars": [self.composition.pressure, self.composition.enthalpy],
            "var_names": [self.composition._pressure_var, self.composition._enthalpy_var]
        }
        for substance in self.composition.substances:
            self.primary_subsystem["vars"].append(substance.overall_fraction)
            self.primary_subsystem["var_names"].append(substance.overall_fraction_var)

        # Use the managers from the composition so that the Schur complement can be made
        self.eqm: pp.ad.EquationManager = self.composition.eq_manager
        self.dofm: pp.DofManager = self.composition.dof_manager

        self.eqm.equations.update({
            "overall_component_fraction_sum": self.composition.overall_component_fractions_sum()
        })

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

        ## model-specific input.
        self._source_quantity = {
            "H20_iapws": 55555.5, # mol in 1 m^3 (1 mol of liquid water is approx 1.8xe-5 m^3)
            "NaCl": 0. # only water is pumped into the system
        }
        self._source_enthalpy = {
            "H20_iapws": 10., # TODO get physical value here
            "NaCl": 0. # no new salt enters the system
        }
        # 110°C temperature for D-BC for conductive flux
        self._conductive_boundary_temperature = 383.15
        # flux for N-BC
        self._outflow_flux = 4000. # if less mole flow out than pumped in, pressure rises
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
        self._set_mass_balance_equations()
        self._set_energy_balance_equation() 

        # NOTE this can be optimized. Some component need to be discretized only once.
        self.eqm.discretize(self.gb)

#------------------------------------------------------------------------------
### SIMULATION related methods and implementation of abstract methods
#------------------------------------------------------------------------------

    def before_newton_loop(self) -> None:
        """Resets the iteration counter and convergence status."""
        self.convergence_status = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Does nothing currently."""
        pass

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """
        Distributes solution of iteration additively to the iterate state of the variables.
        Increases the iteration counter.

        :param solution_vector: solution to global linear system of current iteration
        :type solution_vector: numpy.ndarray
        """
        self._nonlinear_iteration += 1
        self.dofm.distribute_variable(
            values=solution_vector,
            variables=self.primary_subsystem["vars"],
            additive=True, to_iterate=True
        )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Distributes the values from the iterate state to the the state (for next time step).
        Exports the results.
        """
        # solution = self.dofm.assemble_variable(
        #     variables=self.primary_subsystem["vars"], from_iterate=True
        # )

        self.dofm.distribute_variable(solution, variables=self.primary_subsystem["vars"])
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Does nothing currently."""
        pass

    def after_simulation(self) -> None:
        """Does nothing currently."""
        pass

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """APerforms a Newton step for the whole system in a monolithic way, by constructing
        a Schur complement using the equilibrium equations and non-primary variables.
        
        :return: If converged, returns the solution. Until then, returns the update.
        :rtype: numpy.ndarray
        """

        # TODO Eirik mentioned an inverter for block-diagonal systems?
        inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

        # Schur complement elimination of non-primary variables and equations
        A_red, b_red = self.eqm.assemble_schur_complement_system(
            primary_equations=self.primary_subsystem["equations"],
            primary_variables=self.primary_subsystem["vars"],
            inverter=inverter,
        )
        res_norm = np.linalg.norm(b_red)
        if res_norm < tol:
            self.convergence_status = True
            x = self.dofm.assemble_variable(
                    variables=self.primary_subsystem["vars"], from_iterate=True
                )
            return x

        # TODO direct solver only nice to small systems
        dx = spla.spsolve(A_red, b_red)

        # Prolongation of primary variables to global dimensions
        # prolongation = self._prolongation_matrix(self.primary_subsystem["vars"])
        # x = prolongation * x

        return dx

    def solve_equilibrium(
        self, max_iter: Optional[int] = 100, tol: Optional[float] = 1e-10
        ) -> bool:
        """Starts equilibrium computations and consequently flash calculations.
            1. Equilibrium
            2. Saturation Flash
            3. Isenthalpic Flash

        Prints a message if one of them fails to converge.

        :param max_iter: maximal number of iterations for Newton solver
        :type max_iter: int
        :param tol: tolerance for Newton residual
        :type tol: float

        :return: True if successful, False otherwise
        :rtype: bool
        """
        equilibrium = self.composition.compute_phase_equilibrium(max_iter, tol)
        saturations = self.composition.saturation_flash(max_iter, tol)
        isenthalpic = self.composition.isenthalpic_flash(max_iter, tol)

        if equilibrium and saturations and isenthalpic:
            return True
        else:
            if not equilibrium:
                print("Equilibrium Calculations did not converge.")
            if not saturations:
                print("Saturation Flash did not converge.")
            if not isenthalpic:
                print("Isenthalpic Flash did not converge.")
            return False

    def _prolongation_matrix(self, variables: List["pp.ad.MergedVariable"]) -> sps.spmatrix:
        """ Returns the prolongation matrix for mapping the reduced variable
        (by Schur complement) to global dimensions.
        Credits to EK.
        """
        # Construct mappings from subsets of variables to the full set.
        nrows = self.dofm.num_dofs()
        rows = np.unique(
            np.hstack(
                # The use of private variables here indicates that something is wrong
                # with the data structures. Todo..
                [
                    self.dofm.grid_and_variable_to_dofs(s._g, s._name)
                    for var in variables
                    for s in var.sub_vars
                ]
            )
        )
        ncols = rows.size
        cols = np.arange(ncols)
        data = np.ones(ncols)

        return sps.coo_matrix((data, (rows, cols)), shape=(nrows, ncols)).tocsr()

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
        
        for g, d in self.gb:
            
            source = self._unitary_source(g)
            unit_tensor = pp.SecondOrderTensor(1.)
            # TODO is this a must-have parameter?
            zero_vector_source = np.zeros((self.gb.dim_max(), g.num_cells))

            ### Mass weight parameter. Same for all balance equations
            pp.initialize_data(
                g, d, self.mass_parameter_key, {"mass_weight": self._base_porosity
                }
            )

            ### Mass parameters per substance
            
            for i, substance in enumerate(self.composition.substances):
                pp.initialize_data(
                    g,
                    d,
                    "%s_%s" % (self.mass_parameter_key, substance.name),
                    {
                        "source": np.copy(source * self._source_quantity[i]),  
                    },
                )

            ### Darcy flow parameters per PhaseField
            # TODO VL is this necessary per phase? global pressure formulation
            bc, bc_vals = self._bc_advective_flux(g)
            transmissibility = pp.SecondOrderTensor(
                self._base_permeability
            )
            for i, phase in enumerate(self.composition):
                # parameters for MPFA
                pp.initialize_data(
                    g,
                    d,
                    "%s_%s" % (self.flow_parameter_key, phase.name),
                    {
                        "bc": bc,
                        "bc_values": bc_vals,
                        "second_order_tensor": transmissibility,
                        "vector_source": np.copy(zero_vector_source.ravel("F")),
                        "ambient_dimension": self.gb.dim_max(),
                    },
                )
                # parameters for upwinding
                pp.initialize_data(
                    g,
                    d,
                    "%s_%s" % (self.upwind_parameter_key, phase.name),
                    {
                        "bc": bc,
                        "bc_values": bc_vals,
                        "darcy_flux": np.zeros(g.num_faces) # Upwinding expects an initial flux
                    },
                )

            ### Energy parameters for global energy equation
            bc, bc_vals = self._bc_conductive_flux(g)
            # enthalpy sources due to substance mass source
            param_dict = dict()
            for i, substance in enumerate(self.composition.substances):
                param_dict.update({
                    "source_%s" % (substance.name): np.copy(source * self._source_enthalpy[i])
                })
            # other enthalpy sources e.g., hot skeleton
            param_dict.update({
                "source": np.copy(source * 0.)
            })
            # MPFA parameters for conductive BC
            param_dict.update({
                "bc": bc,
                "bc_values": bc_vals,
                "second_order_tensor": unit_tensor,
                "vector_source": np.copy(zero_vector_source.ravel("F")),
                "ambient_dimension": self.gb.dim_max(),
            })
            pp.initialize_data(
                    g,
                    d,
                    self.energy_parameter_key,
                    param_dict,
                )
            # parameters for upwinding
            pp.initialize_data(
                    g,
                    d,
                    "%s_%s" % (self.upwind_parameter_key, self.energy_parameter_key),
                    {
                        "bc": bc,
                        "bc_values": bc_vals,
                    },
                )

        # For now we consider only a single domain
        for e, data_edge in self.gb.edges():
            raise NotImplementedError("Mixed dimensional case not yet available.")

    ### Boundary Conditions

    def _bc_advective_flux(self, g: "pp.Grid") -> Tuple[pp.BoundaryCondition, np.ndarray]:
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
        bc, vals = self._bc_unitary_flux(g, "east", "neu")
        return (bc, vals * self._outflow_flux)
    
    def _bc_diff_disp_flux(self, g: "pp.Grid") -> Tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for diffusive-dispersive flux (Darcy). Override for modifications.

        Phys. Dimensions of FICK's LAW OF DIFFUSION:
            - Dirichlet conditions: [-] (molar, fractional: constant concentration at boundary) 
            - Neumann conditions: [mol / m^2 s] 
              (same as advective flux)
        """
        bc, vals = self._bc_unitary_flux(g, "east", "neu")
        return (bc, vals * self._outflow_flux)
    
    def _bc_conductive_flux(self, g: "pp.Grid") -> Tuple[pp.BoundaryCondition, np.ndarray]:
        """ Conductive BC for Fourier flux in energy equation. Override for modifications.
        
        Phys. Dimensions of CONDUCTIVE HEAT FLUX:
            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [J m^3 / m^2 s] (density * specific enthalpy * Darcy flux)
              (same as convective enthalpy flux)
        """
        bc, vals = self._bc_unitary_flux(g, "south", "dir")
        return (bc, vals * self._conductive_boundary_temperature)

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
    ### MODEL EQUATIONS: Mass and Energy Balance
    #------------------------------------------------------------------------------------------           

    def _set_mass_balance_equations(self) -> None:
        """Set mass balance equations per substance"""

        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, self._grids)
        cp = self.composition
        div = pp.ad.Divergence(grids=self._grids, name="Divergence")

        for subst in self.composition.substances:
            # accumulation term
            accumulation = (
                mass.mass / self.dt * (
                    subst.overall_fraction * cp.composit_density() -
                    subst.overall_fraction.previous_timestep() *
                    cp.composit_density(prev_time=True)
                )
            )

            # advection per phase
            advection = list()
            for phase in cp.phases_of_substance(subst):

                # Advective term due to pressure potential per phase in which subst is present
                kw = "%s_%s" % (self.flow_parameter_key, phase.name)
                darcy_flux = self._mpfa_flux(kw, cp.pressure)
                # TODO add rel perm
                darcy_scalar = (
                    phase.molar_density(cp.pressure, cp.enthalpy) *
                    subst.fraction_in_phase(phase) /
                    phase.viscosity(cp.pressure, cp.enthalpy)
                )

                adv_p = self._upwind(kw, darcy_scalar, darcy_flux, "neu")

                advection.append(adv_p)

            # total advection
            advection = sum(advection)

            # rhs source term
            keyword = "%s_%s" % (self.mass_parameter_key, subst.name)
            source = pp.ad.ParameterArray(keyword, "source", grids=self._grids)

            ### MASS BALANCE PER COMPONENT
            # TODO check minus in front of div terms
            equ_subst = accumulation - div * advection - source
            equ_name  = "mass_balance_%s" % (subst.name)
            self.eqm.equations.update({
                equ_name: equ_subst
            })
            self.primary_subsystem["equations"].append(equ_name)

    def _set_energy_balance_equation(self) -> None:
        """Sets the global energy balance equation in terms of enthalpy."""

        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, self._grids)
        cp = self.composition
        div = pp.ad.Divergence(grids=self._grids, name="Divergence")

        # accumulation term
        accumulation = (
            mass.mass / self.dt * (
                cp.enthalpy * cp.composit_density() -
                cp.enthalpy.previous_timestep() * cp.composit_density(True)
            )
        )

        # advection per phase
        advection = list()
        for phase in cp:

            # Advective term due to pressure potential per phase
            kw = "%s_%s" % (self.flow_parameter_key, phase.name)
            darcy_flux = self._mpfa_flux(kw, cp.pressure)
            # TODO add rel perm
            darcy_scalar = (
                cp.composit_density() *
                phase.enthalpy(cp.pressure, cp.enthalpy) /
                phase.viscosity(cp.pressure, cp.enthalpy)
            )

            adv_p = self._upwind(kw, darcy_scalar, darcy_flux, "neu")

            advection.append(adv_p)

        # total advection
        advection = sum(advection)

        # conduction term
        heat_scalar = list()
        for phase in cp:
            heat_scalar.append(
                phase.saturation * phase.thermal_conductivity(cp.pressure, cp.enthalpy)
            )
        heat_scalar = mass * sum(heat_scalar)

        heat_flux = self._mpfa_flux(self.energy_parameter_key, cp.temperature)

        kw = "%s_%s" % (self.upwind_parameter_key, self.energy_parameter_key)
        conduction = self._upwind(kw, heat_scalar, heat_flux, "dir")

        # source term
        # rock enthalpy source
        source = pp.ad.ParameterArray(self.energy_parameter_key, "source", grids=self._grids)
        # enthalpy source due to mass source
        for subst in cp.substances:
            kw = "source_%s" % (subst.name)
            source += pp.ad.ParameterArray(kw, "source", grids=self._grids)

        ### GLOBAL ENERGY
        # TODO check minus in front of div terms
        equ_subst = accumulation - div * advection - div * conduction - source
        equ_name  = "energy_balance"
        self.eqm.equations.update({
            equ_name: equ_subst
        })
        self.primary_subsystem["equations"].append(equ_name)
    
    def _mpfa_flux(
        self, keyword: str, potential: "pp.ad.MergedVariable"
    ) -> "pp.ad.Operator":
        """MPFA discretization of Darcy potential for given pressure and phase.

        :param keyword: keyword to access data necessary for :class:`~porepy.ad.MpfaAd`
        :type keyword: str
        :param potential: a variable whose gradient causes the flux
        :type potential: :class:`~porepy.ad.MergedVariable`

        :return: MPFA discretization including boundary conditions
        :rtype: :class:`~porepy.ad.Operator`
        """
        mpfa = pp.ad.MpfaAd(keyword, self._grids)
        # bc = pp.ad.ParameterArray(keyword, "bc_values", grids=self._grids)
        bc = pp.ad.BoundaryCondition(keyword, self._grids)
        # TODO mixed BC?
        return mpfa.flux * potential + mpfa.bound_flux * bc

    def _upwind(
        self, keyword: str, scalar: "pp.ad.Operator", flux: "pp.ad.Operator", bc_type: str
    ) -> "pp.ad.Operator":
        """Applies upwinding to a combination of scalar and flux term in AD.

        :param keyword: keyword to access data necessary for :class:`~porepy.ad.MpfaAd`
        :type keyword: str
        :param scalar: scalar part of the flux term
        :type scalar: :class:`~porepy.ad.Operator`
        :param scalar: vectorial part of the flux term
        :type scalar: :class:`~porepy.ad.Operator`

        :return: AD representation of the upwinded flux term
        :rtype: :class:`~porepy.ad.Operator`
        """
        bc_type = str(bc_type)
        upwind = pp.ad.UpwindAd(keyword, self._grids)
        # upwind_T_bc = pp.ad.ParameterArray(kw, "bc_values", self._grids)
        upwind_bc = pp.ad.BoundaryCondition(keyword, self._grids)

        result = (upwind.upwind * scalar) * flux 

        # TODO mixed BC?
        if bc_type == "neu":
            result += upwind.bound_transport_neu * upwind_bc
        elif bc_type == "dir":
            result += upwind.bound_transport_dir * upwind_bc
        else:
            raise NotImplementedError("Unknown BC type '%s' for upwinding." % (bc_type))

        return result
    #------------------------------------------------------------------------------------------
    ### CONSTITUTIVE LAWS
    #------------------------------------------------------------------------------------------

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
