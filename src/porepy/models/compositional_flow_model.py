"""Contains a general composit flow class without reactions

The grid, expected Phases and components can be modified in respective methods.

"""

from __future__ import annotations

from typing import Optional
from iapws import IAPWS95

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
        # time step size
        self.dt: float = 0.5
        self.params: dict = params
        # kPa
        self.initial_pressure = 101.3200  # 1 atm
        # Kelvin
        self.initial_temperature = 323.15  # 50 dec Celsius
        # %
        self.initial_salt_concentration = 0.02
        # kg to mol (per second)
        self.injected_moles_water = 5000.0 / pp.composite.IAPWS95_H2O.molar_mass()
        # initial enthalpy through pressure in MPa and at 30 deg Celsius
        water = pp.composite.IncompressibleFluid("")
        h = water.specific_enthalpy(self.initial_pressure, 303.16)
        self.injected_water_enthalpy = h * self.injected_moles_water
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

        # grid refinement
        self._refinement: int = 7
        # create default grid bucket for this model
        self.mdg: pp.MixedDimensionalGrid
        self.create_grid()

        # contains information about the primary system
        self.flow_subsystem: dict[str, list] = dict()

        # Parameter keywords
        self.flow_keyword: str = "flow"
        self.flow_upwind_parameter_key: str = "upwind_%s" % (self.flow_keyword)
        self.mass_keyword: str = "mass"
        self.energy_keyword: str = "energy"
        self.conduction_upwind_parameter_key: str = "upwind_%s" % (
            self.energy_keyword
        )

        # references to discretization operators
        # they will be set during `prepare_simulation`
        self.mass_matrix: pp.ad.MassMatrixAd
        self.div: pp.ad.Divergence
        self.darcy_flux: pp.ad.MpfaAd
        self.darcy_upwind: pp.ad.UpwindAd
        self.darcy_upwind_bc: pp.ad.BoundaryCondition
        self.conductive_flux: pp.ad.MpfaAd
        self.conductive_upwind: pp.ad.UpwindAd
        self.conductive_upwind_bc: pp.ad.BoundaryCondition

        ### COMPOSITION SETUP
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
        Overwrites/sets the instance variables 'mdg'.
        """
        phys_dims = [3, 1]
        n_cells = [i * self._refinement for i in phys_dims]
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
        """
        # creating relevant components
        water = pp.composite.IAPWS95_H2O(self.ad_sys)
        salt = pp.composite.NaCl(self.ad_sys)
        # creating composition class
        self.composition = pp.composite.Composition(self.ad_sys)
        # creating expected phases
        brine = pp.composite.IncompressibleFluid("brine", self.ad_sys)
        vapor = pp.composite.IdealGas("vapor", self.ad_sys)
        # adding components to phases
        brine.add_component([water, salt])
        vapor.add_component(water)

        # adding everything to the composition class
        self.composition.add_component(water)
        self.composition.add_component(salt)
        self.composition.add_phase(brine)
        self.composition.add_phase(vapor)

        # defining an equilibrium equation for water
        k_value = 1.1
        # k_value = self._P_vap(self.composition.T)
        name = "k_value_" + water.name
        equilibrium = (
            vapor.ext_fraction_of_component(water)
            - k_value * brine.ext_fraction_of_component(water)
        )

        self.composition.add_equilibrium_equation(water, equilibrium, name)

        # setting of initial values
        nc = self.mdg.num_subdomain_cells()
        # setting water feed fraction
        water_frac = (1. - self.initial_salt_concentration) * np.ones(nc)
        self.ad_sys.set_var_values(water.fraction_name, water_frac, True)
        # setting salt feed fraction
        salt_frac = self.initial_salt_concentration * np.ones(nc)
        self.ad_sys.set_var_values(salt.fraction_name, salt_frac, True)
        # setting initial pressure
        p_vals = self.initial_pressure * np.ones(nc)
        self.ad_sys.set_var_values(self.composition.p_name, p_vals, True)
        # setting initial temperature
        T_vals = self.initial_temperature * np.ones(nc)
        self.ad_sys.set_var_values(self.composition.T_name, T_vals, True)
        # set zero enthalpy values at the beginning to get the AD framework properly started
        h_vals = np.zeros(nc)
        self.ad_sys.set_var_values(self.composition.h_name, h_vals, True)

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

        # Exporter initialization for saving results
        self._exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
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
            primary_vars.remove(phase.saturation_name)
            self._satur_vars.append(phase.saturation_name)
        self.flow_subsystem.update({"primary_vars": primary_vars})
        self.flow_subsystem.update({"secondary_vars": secondary_vars})

        # deep copies, just in case
        sec_eq = [eq for eq in self.composition.ph_subsystem["equations"]]
        self.flow_subsystem.update({"secondary_equations": sec_eq})

        self._set_up()
        self._set_feed_fraction_unity_equation()
        self._set_mass_balance_equations()
        self._set_energy_balance_equation()

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
            self._exporter.write_vtu(variables, time_dependent=True)

    ### NEWTON --------------------------------------------------------------------------------

    def before_newton_loop(self) -> None:
        """Resets the iteration counter and convergence status."""
        self.converged = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Re-discretizes the Upwind operators and the fluxes."""
        # MPFA flux upwinding
        # compute the advective flux (grad P)
        kw = self.flow_keyword
        pp.fvutils.compute_darcy_flux(self.mdg, kw, kw, p_name=self.composition.p_name)
        # compute the conductive flux (grad T)
        kw = self.energy_keyword
        pp.fvutils.compute_darcy_flux(self.mdg, kw, kw, p_name=self.composition.T_name)
        ## re-discretize the upwinding of the Darcy flux
        # self.darcy_upwind.upwind.discretize(self.mdg)
        # self.darcy_upwind.bound_transport_dir.discretize(self.mdg)
        # self.darcy_upwind.bound_transport_neu.discretize(self.mdg)
        ## re-discretize the upwinding of the conductive flux
        # self.conductive_upwind.upwind.discretize(self.mdg)
        # self.conductive_upwind.bound_transport_dir.discretize(self.mdg)
        # self.conductive_upwind.bound_transport_neu.discretize(self.mdg)
        for eq in self.ad_sys._equations.values():
            eq.discretize(self.mdg)

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
        if hasattr(self, "_exporter"):
            self._exporter.write_pvd()

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
        # print("Res norm ", res_norm)
        # print("Condition ", np.linalg.cond(A.todense()))
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
            zero_vector_source = np.zeros((self.mdg.dim_max(), g.num_cells))

            ### Mass parameters.
            param_dict = dict()
            # weight for accumulation term
            param_dict.update({"mass_weight": self.porosity * np.ones(g.num_cells)})
            # source terms per component
            for component in self.composition.components:
                param_dict.update({
                    f"source_{component.name}": np.copy(
                        source * self.mass_sources[component.name]
                    )
                })

            pp.initialize_data(
                g,
                d,
                self.mass_keyword,
                param_dict,
            )

            ### Darcy flow parameters for assumed single pressure
            bc, bc_vals = self._bc_advective_flux(g)
            transmissibility = pp.SecondOrderTensor(
                self.permeability * np.ones(g.num_cells)
            )
            # parameters for DARCY FLUX
            pp.initialize_data(
                g,
                d,
                self.flow_keyword,
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "second_order_tensor": transmissibility,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                    "darcy_flux": np.zeros(g.num_faces), # Upwind needs flux values non faces
                },
            )

            ### Energy parameters for global energy equation
            bc, bc_vals = self._bc_conductive_flux(g)
            param_dict = dict()
            # general enthalpy sources e.g., hot skeleton
            param_dict.update({"source": np.copy(source) * 0.0})
            # enthalpy sources due to substance mass source
            for component in self.composition.components:
                param_dict.update(
                    {
                        f"source_{component.name}": np.copy(
                            source * self.enthalpy_sources[component.name]
                        )
                    }
                )

            # CONDUCTIVE FLUX and respective upwinding parameters
            param_dict.update(
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "second_order_tensor": unit_tensor,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                    "darcy_flux": np.zeros(g.num_faces),  # Upwind needs flux values on faces
                }
            )
            pp.initialize_data(
                g,
                d,
                self.energy_keyword,
                param_dict,
            )

        # For now we consider only a single domain
        for e, data_edge in self.mdg.interfaces(return_data=True):
            raise NotImplementedError("Mixed dimensional case not yet available.")

        ### Instantiating discretization operators
        # mass matrix
        self.mass_matrix = pp.ad.MassMatrixAd(self.mass_keyword, self._grids)
        # divergence (based on grid)
        self.div = pp.ad.Divergence(subdomains=self._grids, name="Divergence")
        # darcy flux
        mpfa = pp.ad.MpfaAd(self.flow_keyword, self._grids)
        bc = pp.ad.BoundaryCondition(self.flow_keyword, self._grids)
        self.darcy_flux = mpfa.flux * self.composition.p + mpfa.bound_flux * bc
        # darcy upwind
        self.darcy_upwind = pp.ad.UpwindAd(self.flow_keyword, self._grids)
        self.darcy_upwind_bc = pp.ad.BoundaryCondition(self.flow_keyword, self._grids)

        # conductive flux
        mpfa = pp.ad.MpfaAd(self.energy_keyword, self._grids)
        bc = pp.ad.BoundaryCondition(self.energy_keyword, self._grids)
        self.conductive_flux = (
            mpfa.flux * self.composition.T + mpfa.bound_flux * bc
        )
        # conductive upwind
        self.conductive_upwind = pp.ad.UpwindAd(self.energy_keyword, self._grids)
        self.conductive_upwind_bc = pp.ad.BoundaryCondition(self.energy_keyword, self._grids)

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
        point_source = np.array([[0.5], [0.5]])
        source_cell = g.closest_cell(point_source)
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

        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        self.ad_sys.set_equation(name, equation, num_equ_per_dof=image_info)

    def _set_mass_balance_equations(self) -> None:
        """Set mass balance equations per substance"""
        cp = self.composition

        for component in cp.components:
            ### ACCUMULATION
            accumulation = self.mass_matrix.mass * (
                component.fraction * cp.density()
                - component.fraction.previous_timestep()
                * cp.density(prev_time=True)
            )

            # ADVECTION
            advection_scalar = list()
            for phase in cp.phases_of(component):

                scalar_part = (
                    phase.density(cp.p, cp.T)
                    * phase.ext_fraction_of_component(component)  # TODO use regular composition (debug factorization error when using this)
                    * self.rel_perm(phase.saturation)  # TODO change rel perm access
                    / phase.dynamic_viscosity(cp.p, cp.T)
                )
                advection_scalar.append(scalar_part)
            advection_scalar = sum(advection_scalar)

            advection = (
                self.darcy_flux * (self.darcy_upwind.upwind * advection_scalar)
                - self.darcy_upwind.bound_transport_dir * self.darcy_flux
                * self.darcy_upwind_bc
                - self.darcy_upwind.bound_transport_neu * self.darcy_upwind_bc
            )

            ### SOURCE
            keyword = f"source_{component.name}"
            source = pp.ad.ParameterArray(self.mass_keyword, keyword, subdomains=self._grids)
            source = self.mass_matrix.mass * source

            ### MASS BALANCE PER COMPONENT
            # minus in advection already included
            equation = accumulation + self.dt * (self.div * advection - source)
            equ_name = "mass_balance_%s" % (component.name)
            image_info = dict()
            for sd in self.mdg.subdomains():
                image_info.update({sd: {"cells": 1}})
            self.ad_sys.set_equation(equ_name, equation, num_equ_per_dof=image_info)
            self.flow_subsystem["primary_equations"].append(equ_name)

    def _set_energy_balance_equation(self) -> None:
        """Sets the global energy balance equation in terms of enthalpy."""

        # creating operators, parameters and shorter namespaces
        cp = self.composition
        upwind_adv = self.darcy_upwind
        upwind_adv_bc = self.darcy_upwind_bc
        upwind_cond = self.conductive_upwind
        upwind_cond_bc = self.conductive_upwind_bc

        ### ACCUMULATION
        accumulation = self.mass_matrix.mass * (
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
            - upwind_adv.bound_transport_dir * self.darcy_flux * upwind_adv_bc
            - upwind_adv.bound_transport_neu * upwind_adv_bc
        )

        ### CONDUCTION
        conductive_scalar = list()
        for phase in cp.phases:
            conductive_scalar.append(
                phase.saturation
                * phase.thermal_conductivity(cp.p, cp.T) * 10.  # TODO remove 100 (debug)
            )
        conductive_scalar = sum(conductive_scalar)

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
        self.ad_sys.set_equation(equ_name, equation, num_equ_per_dof=image_info)
        self.flow_subsystem["primary_equations"].append(equ_name)

    ### CONSTITUTIVE LAWS ---------------------------------------------------------------------

    def rel_perm(self, saturation: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Helper function until data structure for heuristic laws is done."""
        return saturation * saturation
