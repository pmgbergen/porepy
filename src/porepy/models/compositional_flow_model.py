"""Contains a general composit flow class without reactions

The grid, expected Phases and components can be modified in respective methods.

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plot
import matplotlib.colors as mcolors

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
        # kPa
        self.initial_pressure = 101.3200
        # Kelvin
        self.initial_temperature = 323.15  # 50 dec Celsius
        # %
        self.initial_salt_concentration = 0.02
        # kg to mol
        self.injected_moles_water = 10.0 / pp.composite.IAPWS95_H2O.molar_mass()
        # kJ / mol , specific heat capacity from Wikipedia
        # rough approximation of enthalpy source from injected material
        # injection at ca 30 deg Celsius and 2x atmospheric pressue
        self.injected_water_enthalpy = (
            0.075327 * 300
            + 2 * self.initial_pressure / (998.21 / pp.composite.IAPWS95_H2O.molar_mass())
        )
        # D-BC temperature Kelvin for conductive flux
        self.boundary_temperature = 423.15  # 150 Celsius
        # D-BC on outflow boundary
        self.boundary_pressure = self.initial_pressure
        # base porosity for grid
        self.porosity = 1.0
        # base permeability for grid
        self.permeability = 1.0
        # solver strategy. If monolithic, the model will take care of flash calculations
        # if not, it will solve only the primary system
        self.monolithic = monolithic_solver

        # time step size
        self.dt: float = 0.5
        # residual tolerance for the balance equations
        self.tolerance_balance_equations = 1e-8
        # create default grid bucket for this model
        self.mdg: pp.MixedDimensionalGrid
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
        self.div: pp.ad.Divergence
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

        # Use the managers from the composition and add the balance equations
        self.ad_sys: pp.ad.ADSystem = pp.ad.ADSystem(self.mdg)
        self.dof_man: pp.DofManager = self.ad_sys.dof_manager
        self.set_composition()
        # model specific sources, this must be modified for every composition
        self.mass_sources = {
            "IAPWS95_H2O": self.injected_moles_water,  
            "NaCl": 0.0,  # only water is pumped into the system
        }
        self.enthalpy_sources = {
            "IAPWS95_H2O": self.injected_water_enthalpy,  # in kJ
            "NaCl": 0.0,  # no new salt enters the system
        }
        # indicator if converged
        self.converged: bool = False
        ### PRIVATE

        # projections/ prolongation from global dofs to primary and secondary variables
        # will be set during `prepare simulations`
        self._prolong_prim: sps.spmatrix
        self._prolong_sec: sps.spmatrix
        self._prolong_system: sps.spmatrix
        # system variables, depends whether solver monolithic or not
        self._system_vars: list
        # exporter
        self._exporter: pp.Exporter
        # list of grids as ordered in GridBucket
        self._grids = [g for g in self.mdg.subdomains()]
        # list of edges as ordered in GridBucket
        self._edges = [e for e in self.mdg.interfaces()]
        # storing names of all saturation variables, which are tertiary
        self._satur_vars: list[str] = list()
        # data for Schur complement expansion
        self._for_expansion = (None, None, None)

    def create_grid(self) -> None:
        """Assigns a cartesian grid as computational domain.
        Overwrites the instance variables 'gb'.
        """
        cells_per_dim = 3
        phys_dims = [1, 1]
        n_cells = [i * cells_per_dim for i in phys_dims]
        bounding_box_points = np.array([[0, phys_dims[0]],[0, phys_dims[1]]])
        self.box = pp.geometry.bounding_box.from_points(bounding_box_points)
        sg = pp.CartGrid(n_cells, phys_dims)
        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(sg)
        self.mdg.compute_geometry()

    def set_composition(self) -> None:
        """Define the composition for which the simulation should be run and performs
        the initial (p-T) equilibrium calculations for the initial state given in p-T and feed.

        Use this method to inherit and override the composition, while keeping the (generic)
        rest of the model.
        """
        # creating relevant components
        self.water = pp.composite.IAPWS95_H2O(self.ad_sys)
        self.salt = pp.composite.NaCl(self.ad_sys)
        # creating composition class
        self.composition = pp.composite.Composition(self.ad_sys)
        # creating expected phases
        self.brine = pp.composite.IncompressibleFluid("brine", self.ad_sys)
        self.vapor = pp.composite.IdealGas("vapor", self.ad_sys)
        # adding components to phases
        self.brine.add_component([self.water, self.salt])
        self.vapor.add_component(self.water)

        # adding everything to the composition class
        self.composition.add_component(self.water)
        self.composition.add_component(self.salt)
        self.composition.add_phase(self.brine)
        self.composition.add_phase(self.vapor)

        # defining an equilibrium equation for water
        k_value = 0.9
        name = "k_value_" + self.water.name
        equilibrium = (
            self.vapor.component_fraction_of(self.water)
            - k_value * self.brine.component_fraction_of(self.water)
        )

        self.composition.add_equilibrium_equation(self.water, equilibrium, name)

        self.composition.initialize()

        # setting of initial values
        nc = self.mdg.num_subdomain_cells()
        # setting water feed fraction
        water_frac = (1. - self.initial_salt_concentration) * np.ones(nc)
        self.ad_sys.set_var_values(self.water.fraction_var_name, water_frac, True)
        # setting salt feed fraction
        salt_frac = self.initial_salt_concentration * np.ones(nc)
        self.ad_sys.set_var_values(self.salt.fraction_var_name, salt_frac, True)
        # setting initial pressure
        p_vals = self.initial_pressure * np.ones(nc)
        self.ad_sys.set_var_values(self.composition._p_var, p_vals, True)
        # setting initial temperature
        T_vals = self.initial_temperature * np.ones(nc)
        self.ad_sys.set_var_values(self.composition._T_var, T_vals, True)
        # set zero enthalpy values at the beginning to get the AD framework properly started
        h_vals = np.zeros(nc)
        self.ad_sys.set_var_values(self.composition._h_var, h_vals, True)

        self.composition.isothermal_flash(copy_to_state=True, initial_guess="feed")
        self.composition.evaluate_saturations()
        # This corrects the initial values
        self.composition.evaluate_specific_enthalpy()
        
    def prepare_simulation(self) -> None:
        """Preparing essential simulation configurations.

        Method needs to be called after the composition has been set and prior to applying any
        solver.

        """

        # Exporter initialization for saving results
        self._exporter = pp.Exporter(
            self.mdg, self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

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
        # deep copies, just in case
        primary_vars = [var for var in self.composition.ph_subsystem["secondary_vars"]]
        secondary_vars = [var for var in self.composition.ph_subsystem["primary_vars"]]
        self._satur_vars = list()
        # remove the saturations, which are actually secondary in both systems... tertiary?TODO
        for phase in self.composition.phases:
            primary_vars.remove(phase.saturation_var_name)
            self._satur_vars.append(phase.saturation_var_name)
        self.flow_subsystem.update({"primary_vars": primary_vars})
        self.flow_subsystem.update({"secondary_vars": secondary_vars})

        # deep copies, just in case
        sec_eq = [eq for eq in self.composition.ph_subsystem["equations"]]
        self.flow_subsystem.update({"secondary_equations": sec_eq})

        self._set_up()
        self._set_feed_fraction_unity_equation()
        self._set_mass_balance_equations()
        self._set_energy_balance_equation()

        # NOTE this can be optimized. Some component need to be discretized only once.
        self.ad_sys.discretize()
        # prepare datastructures for the solver
        self._prolong_prim = self.dof_man.projection_to(primary_vars).transpose()
        self._prolong_sec = self.dof_man.projection_to(secondary_vars).transpose()
        
        # the system vars are everything except the saturations
        self._system_vars = (
            self.flow_subsystem["primary_vars"]
            + self.flow_subsystem["secondary_vars"]
        )
        self._prolong_system = self.dof_man.projection_to(self._system_vars).transpose()
        
        self._export()

    def print_x(self, where=""):
        print("-------- %s" % (str(where)))
        print("Iterate")
        print(self.dof_man.assemble_variable(from_iterate=True))
        print("State")
        print(self.dof_man.assemble_variable(from_iterate=False))

    def matrix_plot(self, J):
        print(J.todense())
        plot.figure()
        plot.subplot(211)
        plot.matshow(J.todense())
        plot.colorbar(orientation="vertical")
        plot.set_cmap("terrain")

        plot.subplot(212)
        norm = mcolors.TwoSlopeNorm(vmin=-10.0, vcenter=0, vmax=10.0)
        plot.matshow(J.todense(),norm=norm,cmap='RdBu_r')
        plot.colorbar()
        
        plot.show()

    def _export(self) -> None:
        if hasattr(self, "_exporter"):
            variables = (
                self.flow_subsystem["primary_vars"]
                + self.flow_subsystem["secondary_vars"]
                + self._satur_vars
            )
            # self._exporter.write_vtu([variables])

    ### NEWTON --------------------------------------------------------------------------------

    def before_newton_loop(self) -> None:
        """Resets the iteration counter and convergence status."""
        self.converged = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Re-discretizes the Upwind operators and the fluxes."""
        # Darcy flux upwinding
        # compute the flux
        kw = self.flow_parameter_key
        kw_store = self.flow_upwind_parameter_key
        variable = self.composition._p_var  # TODO access to var name
        pp.fvutils.compute_darcy_flux(self.mdg, kw, kw_store, p_name=variable)
        # re-discretize the upwinding of the Darcy flux
        self.darcy_upwind.upwind.discretize(self.mdg)
        self.darcy_upwind.bound_transport_dir.discretize(self.mdg)
        self.darcy_upwind.bound_transport_neu.discretize(self.mdg)

        # compute the heat flux
        kw = self.energy_parameter_key
        kw_store = self.conduction_upwind_parameter_key
        variable = self.composition._T_var  # TODO access to var name
        pp.fvutils.compute_darcy_flux(self.mdg, kw, kw_store, p_name=variable)
        # re-discretize the upwinding of the conductive flux
        self.conductive_upwind.upwind.discretize(self.mdg)
        self.conductive_upwind.bound_transport_dir.discretize(self.mdg)
        self.conductive_upwind.bound_transport_neu.discretize(self.mdg)

        if not self.monolithic:
            print(f".. .. isenthalpic flash at iteration {self._nonlinear_iteration}")
            success = self.composition.isenthalpic_flash(False, initial_guess="iterate")
            if not success:
                self.print_x("Isenthalpic flash failure.")
                raise RuntimeError("FAILURE: Isenthalpic flash.")
            else:
                print(".. .. Success: Isenthalpic flash.")
        self.composition.evaluate_saturations(False)

    def after_newton_iteration(
        self, solution_vector: np.ndarray, iteration: int
    ) -> None:
        """Distributes solution of iteration additively to the iterate state of the variables.
        Increases the iteration counter.
        """
        self._nonlinear_iteration += 1

        if self.monolithic:
            # expant
            DX = self._prolong_system * solution_vector
        else:
            inv_A_ss, b_s, A_sp = self._for_expansion
            x_s = inv_A_ss * (b_s - A_sp * solution_vector)
            DX = self._prolong_prim * solution_vector + self._prolong_sec * x_s

        self.dof_man.distribute_variable(
                values=DX,
                variables=self._system_vars,
                additive=True,
                to_iterate=True,
            )
        self.composition.evaluate_saturations(False)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Distributes the values from the iterate state to the the state
        (for next time step).
        Exports the results.
        """
        self.dof_man.distribute_variable(solution, variables=self._system_vars)
        self.composition.evaluate_saturations(True)
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Reset iterate state to previous state."""
        X = self.dof_man.assemble_variable()
        self.dof_man.distribute_variable(X, to_iterate=True)

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
            A, b = self.ad_sys.assemble_subsystem(variables=self._system_vars)
        else:
            # non-linear Schur complement elimination of secondary variables
            A_pp, b_p = self.ad_sys.assemble_subsystem(
                self.flow_subsystem["primary_equations"],
                self.flow_subsystem["primary_vars"]
            )
            A_sp, _ = self.ad_sys.assemble_subsystem(
                self.flow_subsystem["secondary_equations"],
                self.flow_subsystem["primary_vars"]
            )
            A_ps, _ = self.ad_sys.assemble_subsystem(
                self.flow_subsystem["primary_equations"],
                self.flow_subsystem["secondary_vars"]
            )
            A_ss, b_s = self.ad_sys.assemble_subsystem(
                self.flow_subsystem["secondary_equations"],
                self.flow_subsystem["secondary_vars"]
            )
            if self.composition._last_inverted is not None:
                inv_A_ss = self.composition._last_inverted
            else:
                inv_A_ss = np.linalg.inv(A_ss.A)
            inv_A_ss = sps.csr_matrix(inv_A_ss)
            A = A_pp - A_ps * inv_A_ss * A_sp
            A = sps.csr_matrix(A)
            b = b_p - A_ps * inv_A_ss * b_s
            self._for_expansion = (inv_A_ss, b_s, A_sp)

        res_norm = np.linalg.norm(b)
        if res_norm < tol:
            self.converged = True
            x = self.dof_man.assemble_variable(variables=self._system_vars, from_iterate=True)
            return x

        dx = spla.spsolve(A, b)

        return dx

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

    ### SET-UP --------------------------------------------------------------------------------

    ## collective set-up method
    
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
                            source * self.mass_sources[component.name]
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
                            source * self.enthalpy_sources[component.name]
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

        # divergence (based on grid)
        self.div = pp.ad.Divergence(subdomains=self._grids, name="Divergence")

        # darcy flux
        mpfa = pp.ad.MpfaAd(self.flow_parameter_key, self._grids)
        bc = pp.ad.BoundaryCondition(self.flow_parameter_key, self._grids)
        self.darcy_flux = mpfa.flux * self.composition.p + mpfa.bound_flux * bc

        # darcy upwind
        keyword = self.flow_upwind_parameter_key
        self.darcy_upwind = pp.ad.UpwindAd(keyword, self._grids)
        self.darcy_upwind_bc = pp.ad.BoundaryCondition(keyword, self._grids)

        # conductive flux
        mpfa = pp.ad.MpfaAd(self.energy_parameter_key, self._grids)
        bc = pp.ad.BoundaryCondition(self.energy_parameter_key, self._grids)
        self.conductive_flux = (
            mpfa.flux * self.composition.T + mpfa.bound_flux * bc
        )
        # conductive upwind
        keyword = self.conduction_upwind_parameter_key
        self.conductive_upwind = pp.ad.UpwindAd(keyword, self._grids)
        self.conductive_upwind_bc = pp.ad.BoundaryCondition(keyword, self._grids)

    ## Boundary Conditions

    def _bc_advective_flux(self, g: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC for advective flux (Darcy). Override for modifications.

        Phys. Dimensions of ADVECTIVE MASS FLUX:

            - Dirichlet conditions: [kPa]
            - Neumann conditions: [mol / m^2 s] = [(mol / m^3) * (m^3 / m^2 s)]
                (molar density * Darcy flux)

        Phys. Dimensions of ADVECTIVE ENTHALPY FLUX:

            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [kJ / m^2 s] (density * specific enthalpy * Darcy flux)

        Notes:
            Enthalpy flux D-BCs need some more thoughts. Isn't it unrealistic to assume the
            temperature or enthalpy of the outflowing fluid is known?
            That BC would influence our physical setting and it's actually our goal to find out
            how warm the water will be at the outflow.
        
            BC has to be defined for all fluxes separately.

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
        raise NotImplementedError("Diffusive-Dispersive Boundary Flux not available.")

    def _bc_conductive_flux(
        self, g: pp.Grid
    ) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """Conductive BC for Fourier flux in energy equation. Override for modifications.

        Phys. Dimensions of CONDUCTIVE HEAT FLUX:

            - Dirichlet conditions: [K] (temperature)
            - Neumann conditions: [kJ / m^2 s] (density * specific enthalpy * Darcy flux)
              (same as convective enthalpy flux)

        """
        bc, vals = self._bc_unitary_flux(g, "south", "dir")
        return (bc, vals * self.boundary_temperature)

    def _bc_unitary_flux(
        self, g: pp.Grid, side: str, bc_type: Optional[str] = "neu"
    ) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC objects for unitary flux on specified grid side.

        Parameters:
            g: grid (single-dim domain)
            side ('north', 'east', 'south', 'west'): side of grid with non-zero flux values
            bc_type ('neu', 'dir'): BC types for the non-zero side. By default zero-Neumann BC
                are set anywhere else.

        Returns:
            a tuple containing the BC object and values for the boundary faces of the grid.

        """
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
                f"Unknown grid side '{side}' for unitary flux. Use 'west', 'north',..."
            )

        # non-zero on chosen side, the rest is zero-Neumann by default
        bc = pp.BoundaryCondition(g, idx, bc_type)
        # homogeneous BC
        vals = np.zeros(g.num_faces)
        # unitary on chosen side
        vals[idx] = 1.0

        return (bc, vals)

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
        source_cell = g.closest_cell(np.array([0.5, 0.5]))
        vals[source_cell] = 1.0

        return vals

    ### MODEL EQUATIONS -----------------------------------------------------------------------

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

        self.ad_sys.set_equation(name, equation)

    def _set_mass_balance_equations(self) -> None:
        """Set mass balance equations per substance"""

        # creating operators, parameters and shorter namespaces
        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, self._grids)
        cp = self.composition
        upwind = self.darcy_upwind
        bc_upwind = self.darcy_upwind_bc

        for component in self.composition.components:
            ### ACCUMULATION
            accumulation = mass.mass * (
                component.fraction * cp.density()
                - component.fraction.previous_timestep()
                * cp.density(prev_time=True)
            )

            # ADVECTION
            advection_scalar = list()
            for phase in cp.phases_of(component):

                scalar_part = (
                    phase.density(cp.p, cp.T)
                    * phase.component_fraction_of(component)
                    * self.rel_perm(phase.saturation)  # TODO change rel perm access
                    / phase.dynamic_viscosity(cp.p, cp.T)
                )
                advection_scalar.append(scalar_part)
            advection_scalar = sum(advection_scalar)

            advection = (
                self.darcy_flux * (upwind.upwind * advection_scalar)
                - upwind.bound_transport_dir * self.darcy_flux * bc_upwind
                - upwind.bound_transport_neu * bc_upwind
            )

            ### SOURCE
            keyword = "%s_%s" % (self.mass_parameter_key, component.name)
            source = pp.ad.ParameterArray(keyword, "source", subdomains=self._grids)

            ### MASS BALANCE PER COMPONENT
            # minus in advection already included
            equation = accumulation + self.dt * (self.div * advection - source)
            equ_name = "mass_balance_%s" % (component.name)
            self.ad_sys.set_equation(equ_name, equation)
            self.flow_subsystem["primary_equations"].append(equ_name)

    def _set_energy_balance_equation(self) -> None:
        """Sets the global energy balance equation in terms of enthalpy."""

        # creating operators, parameters and shorter namespaces
        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, self._grids)
        cp = self.composition
        upwind_adv = self.darcy_upwind
        bc_upwind_adv = self.darcy_upwind_bc
        upwind_cond = self.conductive_upwind
        bc_upwind_cond = self.conductive_upwind_bc

        ### ACCUMULATION
        accumulation = mass.mass * (
            cp.h * cp.density() - cp.h.previous_timestep() * cp.density(prev_time=True)
        )

        ### ADVECTION
        advective_scalar = list()
        for phase in cp.phases:
            scalar_part = (
                phase.density(cp.p, cp.T)
                * phase.specific_enthalpy(cp.p, cp.T)
                * self.rel_perm(phase.saturation)  # TODO change rel perm access
                / phase.dynamic_viscosity(cp.p, cp.T)
            )
            advective_scalar.append(scalar_part)
        advective_scalar = sum(advective_scalar)

        advection = (
            self.darcy_flux * (upwind_adv.upwind * advective_scalar)
            - upwind_adv.bound_transport_dir * self.darcy_flux * bc_upwind_adv
            - upwind_adv.bound_transport_neu * bc_upwind_adv
        )

        ### CONDUCTION
        conductive_scalar = list()
        for phase in cp.phases:
            conductive_scalar.append(
                phase.saturation
                * phase.thermal_conductivity(cp.p, cp.T)
            )
        conductive_scalar = mass.mass * sum(conductive_scalar)

        conduction = (
            self.conductive_flux * (upwind_cond.upwind * conductive_scalar)
            - upwind_cond.bound_transport_dir * self.conductive_flux * bc_upwind_cond
            - upwind_cond.bound_transport_neu * bc_upwind_cond
        )

        ### SOURCE
        # rock enthalpy source
        source = pp.ad.ParameterArray(
            self.energy_parameter_key, "source", subdomains=self._grids
        )
        # enthalpy source due to mass source
        for component in cp.components:
            kw = "source_%s" % (component.name)
            source += pp.ad.ParameterArray(
                self.energy_parameter_key, kw, subdomains=self._grids
            )

        ### GLOBAL ENERGY BALANCE
        equation = accumulation + self.dt * (self.div * (advection + conduction) - source)
        equ_name = "energy_balance"
        self.ad_sys.set_equation(equ_name, equation)
        self.flow_subsystem["primary_equations"].append(equ_name)

    ### CONSTITUTIVE LAWS ---------------------------------------------------------------------

    def rel_perm(self, saturation: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Helper function until data structure for heuristic laws is done."""
        return saturation * saturation
