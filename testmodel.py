from __future__ import annotations

import sys
sys.path.append("/mnt/c/Users/vl-work/Desktop/github/porepy/src")

from datetime import datetime

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp


class TestModel(pp.models.abstract_model.AbstractModel):
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

        self._monolithic: bool = params.get("monolithic", True)
        self._use_pressure_equation: bool = params.get("use_pressure_equation", True)

        if self._use_pressure_equation and not self._monolithic:
            raise RuntimeError("Pressure equation only monolithic.")

        self.params: dict = params
        self.converged: bool = False
        self.dt: float = 0.5

        ## Initial Conditions
        self.initial_p: float = 100
        self.initial_sw: float = 0.
        self.initial_sn: float = 1.
        
        self.inflow_p: float = 150
        self.outflow_p: float = 100
        
        self.inflow_sw: float = 0.7
        self.inflow_sn: float = 0.3

        self.rho_w: float = 1.
        self.rho_n: float = 1.
        self.permeability: float = 0.1

        self.mdg: pp.MixedDimensionalGrid
        self.create_grid()

        self.ad_system: pp.ad.ADSystem = pp.ad.ADSystem(self.mdg)
        self.dof_manager: pp.DofManager = self.ad_system.dof_manager

        self.p: pp.ad.MergedVariable = self.ad_system.create_variable("p")
        self.sw: pp.ad.MergedVariable = self.ad_system.create_variable("s_w")
        self.sn: pp.ad.MergedVariable = self.ad_system.create_variable("s_n")

        val = np.ones(self.mdg.num_subdomain_cells())
        self.ad_system.set_var_values("p", self.initial_p * val, True)
        self.ad_system.set_var_values("s_w", self.initial_sw * val, True)
        self.ad_system.set_var_values("s_n", self.initial_sn * val, True)

        # contains information about the primary system
        self.system: dict[str, list] = dict()

        # Parameter keywords
        self.flow_keyword: str = "flow"
        self.upwind_keyword: str = "upwind"
        self.mass_keyword: str = "mass"


        ## References to discretization operators
        # they will be set during `prepare_simulation`
        self.mass_matrix: pp.ad.MassMatrixAd
        self.div: pp.ad.Divergence
        self.mpfa: pp.ad.MpfaAd
        self.p_bc: pp.ad.BoundaryCondition
        self.upwind_sw: pp.ad.UpwindAd
        self.upwind_sw_bc: pp.ad.BoundaryCondition
        self.upwind_sn: pp.ad.UpwindAd
        self.upwind_sn_bc: pp.ad.BoundaryCondition

        ### PRIVATE
        self._prolong_prim: sps.spmatrix
        self._prolong_sec: sps.spmatrix
        self._prolong_system: sps.spmatrix

        self._system_vars: list[str] = list()
        self._export_vars: list[str] = list()

        self._exporter: pp.Exporter = pp.Exporter(
            self.mdg,
            params["file_name"],
            folder_name=params["folder_name"],
            export_constants_separately=False,
        )

        self._grids = [g for g in self.mdg.subdomains()]
        self._edges = [e for e in self.mdg.interfaces()]

        self.test = False

    def create_grid(self) -> None:
        """Assigns a cartesian grid as computational domain.
        Overwrites/sets the instance variables 'mdg'.
        """
        refinement = 15
        phys_dims = [3, 1]
        n_cells = [10, 2]
        # n_cells = [i * refinement for i in phys_dims]
        bounding_box_points = np.array([[0, phys_dims[0]],[0, phys_dims[1]]])
        self.box = pp.geometry.bounding_box.from_points(bounding_box_points)
        sg = pp.CartGrid(n_cells, phys_dims)
        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(sg)
        self.mdg.compute_geometry()

    def prepare_simulation(self) -> None:

        self.system.update(
            {
                "primary_equations": list(),
                "secondary_equations": list(),
                "primary_vars": list(),
                "secondary_vars": list(),
            }
        )
        if self._use_pressure_equation:
            primary_vars = ["p", "s_w"]
            secondary_vars = ["s_n"]
        else:
            if self._monolithic:
                primary_vars = ["p", "s_w", "s_n"]
                secondary_vars = []
            else:
                primary_vars = ["p", "s_w"]
                secondary_vars = ["s_n"]

        self.system.update({"primary_vars": primary_vars})
        self.system.update({"secondary_vars": secondary_vars})
        self._system_vars = primary_vars + secondary_vars

        export_vars = set(self._system_vars)
        self._export_vars = list(export_vars)

        self._prolong_prim = self.dof_manager.projection_to(primary_vars).transpose()
        self._prolong_sec = self.dof_manager.projection_to(secondary_vars).transpose()
        self._prolong_system = self.dof_manager.projection_to(self._system_vars).transpose()

        self._sn_by_unity = self.get_sn()

        self._set_up()

        if self._use_pressure_equation:
            self.set_pressure_equation()
        else:
            self.set_unity()

        self.set_satur_equation()

        self.ad_system.discretize()
        self._export()

    def _export(self) -> None:
        self._exporter.write_vtu(self._export_vars, time_dependent=True)

    def get_sn(self):
        eq = pp.ad.Scalar(1.)
        return eq - self.sw

    ### SET-UP --------------------------------------------------------------------------------
    
    def _set_up(self) -> None:

        for sd, data in self.mdg.subdomains(return_data=True):

            zero_vector_source = np.zeros((self.mdg.dim_max(), sd.num_cells))

            ### MASS PARAMETERS AND MASS SOURCES
            pp.initialize_data(
                sd,
                data,
                self.mass_keyword,
                {"mass_weight": np.ones(sd.num_cells)},
            )

            ### MASS BALANCE EQUATIONS
            # advective flux in mass balance
            bc, bc_vals = self._bc_advective_flux(sd)
            transmissibility = pp.SecondOrderTensor(np.ones(sd.num_cells) * self.permeability)
            pp.initialize_data(
                sd,
                data,
                self.flow_keyword,
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "second_order_tensor": transmissibility,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                    "darcy_flux": np.zeros(sd.num_faces),
                },
            )

            free_flow = self._bc_freeflow(sd)
            bc, bc_vals = self._bc_advective_upwind_sw(sd)
            pp.initialize_data(
                sd,
                data,
                f"{self.upwind_keyword}_sw",
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "darcy_flux": np.zeros(sd.num_faces),
                    "freeflow_bc": free_flow,
                },
            )

            bc, bc_vals = self._bc_advective_upwind_sn(sd)
            pp.initialize_data(
                sd,
                data,
                f"{self.upwind_keyword}_sn",
                {
                    "bc": bc,
                    "bc_values": bc_vals,
                    "darcy_flux": np.zeros(sd.num_faces),
                    "freeflow_bc": free_flow,
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
        bc = pp.ad.BoundaryCondition(self.flow_keyword, self._grids)
        self.mpfa = pp.ad.MpfaAd(self.flow_keyword, self._grids)
        self.p_bc = pp.ad.BoundaryCondition(self.flow_keyword, self._grids)

        kw = f"{self.upwind_keyword}_sw"
        self.upwind_sw = pp.ad.UpwindAd(kw, self._grids)
        self.upwind_sw_bc = pp.ad.BoundaryCondition(kw, self._grids)

        kw = f"{self.upwind_keyword}_sn"
        self.upwind_sn = pp.ad.UpwindAd(kw, self._grids)
        self.upwind_sn_bc = pp.ad.BoundaryCondition(kw, self._grids)

    ## Boundary Conditions

    def _bc_freeflow(self, sd: pp.Grid) -> np.ndarray:
        all_idx, idx_east, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces, dtype=bool)
        vals[idx_east] = True

        return vals

    def _bc_advective_flux(self, sd: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        _, idx_east, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces)
        vals[idx_east] = self.outflow_p

        if self.inflow_p:
            bc = pp.BoundaryCondition(sd, np.logical_or(idx_east, idx_west), "dir")
            vals[idx_west] = self.inflow_p
        else:
            bc = pp.BoundaryCondition(sd, idx_east, "dir")

        return bc, vals

    def _bc_advective_upwind_sw(self, sd: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        all_idx, idx_east, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces)
        vals[idx_west] = self.inflow_sw
        bc = pp.BoundaryCondition(sd, idx_west, "dir")

        return bc, vals

    def _bc_advective_upwind_sn(self, sd: pp.Grid) -> tuple[pp.BoundaryCondition, np.ndarray]:
        """BC values for the scalar part in the advective flux in component mass balance."""
        all_idx, idx_east, idx_west, *_ = self._domain_boundary_sides(sd)
        
        vals = np.zeros(sd.num_faces)
        vals[idx_west] = self.inflow_sn
        bc = pp.BoundaryCondition(sd, idx_west, "dir")

        return bc, vals

    ### NEWTON --------------------------------------------------------------------------------

    def before_newton_loop(self) -> None:
        self.converged = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:

        pp.fvutils.compute_darcy_flux(
            self.mdg,
            self.flow_keyword,
            self.flow_keyword,
            p_name="p",
            from_iterate=True,
        )

        for sd, data in self.mdg.subdomains(return_data=True):
            flux  = data["parameters"][self.flow_keyword]["darcy_flux"]

            data["parameters"][f"{self.upwind_keyword}_sw"]["darcy_flux"] = np.copy(flux)
            data["parameters"][f"{self.upwind_keyword}_sn"]["darcy_flux"] = np.copy(flux)
        
        self.upwind_sw.upwind.discretize(self.mdg)
        # self.upwind_sw.bound_transport_dir.discretize(self.mdg)
        # self.upwind_sw.bound_transport_neu.discretize(self.mdg)

    def after_newton_iteration(
        self, solution_vector: np.ndarray, iteration: int
    ) -> None:

        self._nonlinear_iteration += 1

        if self._use_pressure_equation:
            DX = self._prolong_prim * solution_vector
        else:

            if self._monolithic:
                DX = self._prolong_system * solution_vector
            else:
                inv_A_ss, b_s, A_sp = self._for_expansion
                x_s = inv_A_ss * (b_s - A_sp * solution_vector)
                DX = self._prolong_prim * solution_vector + self._prolong_sec * x_s

        self.dof_manager.distribute_variable(
                values=DX,
                variables=self._system_vars,
                additive=True,
                to_iterate=True,
            )
        if self._use_pressure_equation:
            sn = self._sn_by_unity.evaluate(self.dof_manager).val
            self.ad_system.set_var_values("s_n", sn, False)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:

        self.dof_manager.distribute_variable(solution, variables=self._system_vars)

        if self._use_pressure_equation:
            sn = self._sn_by_unity.evaluate(self.dof_manager).val
            self.ad_system.set_var_values("s_n", sn, True)
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Reset iterate state to previous state."""
        X = self.dof_manager.assemble_variable()
        self.dof_manager.distribute_variable(X, to_iterate=True)

    def after_simulation(self) -> None:
        self._exporter.write_pvd()

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        
        if self._use_pressure_equation:
            A, b = self.ad_system.assemble_subsystem(variables=self.system["primary_vars"])
        else:
            if self._monolithic:
                A, b = self.ad_system.assemble_subsystem(variables=self._system_vars)
            else:
                # non-linear Schur complement elimination of secondary variables
                A_pp, b_p = self.ad_system.assemble_subsystem(
                    self.system["primary_equations"],
                    self.system["primary_vars"]
                )
                A_sp, _ = self.ad_system.assemble_subsystem(
                    self.system["secondary_equations"],
                    self.system["primary_vars"]
                )
                A_ps, _ = self.ad_system.assemble_subsystem(
                    self.system["primary_equations"],
                    self.system["secondary_vars"]
                )
                A_ss, b_s = self.ad_system.assemble_subsystem(
                    self.system["secondary_equations"],
                    self.system["secondary_vars"]
                )

                inv_A_ss = np.linalg.inv(A_ss.A)
                inv_A_ss = sps.csr_matrix(inv_A_ss)

                A = A_pp - A_ps * inv_A_ss * A_sp
                A = sps.csr_matrix(A)
                b = b_p - A_ps * inv_A_ss * b_s
                self._for_expansion = (inv_A_ss, b_s, A_sp)

        if np.linalg.norm(b) < tol:
            self.converged = True
            x = self.dof_manager.assemble_variable(variables=self._system_vars, from_iterate=True)
            return x

        dx = spla.spsolve(A, b)

        return dx

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

    def set_unity(self) -> None:
        """Sets the equation representing the feed fraction unity.

        Performs additionally an index reduction on this algebraic equation.

        """

        name = "unity"
        if self._monolithic:
            self.system["primary_equations"].append(name)
        else:
            self.system["secondary_equations"].append(name)

        unity = pp.ad.Scalar(1.)

        # index reduction of algebraic unitarity constraint
        # demand exponential decay of rate of change
        time_derivative = [
            self.sw - self.sw.previous_timestep()
            +  self.sn - self.sn.previous_timestep()
        ]
        unity -= self.sw
        unity -= self.sn

        decay = self.dt / (2 * np.pi)

        equation = sum(time_derivative) + self.dt * decay * unity

        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        
        self.ad_system.set_equation(name, equation, num_equ_per_dof=image_info)

    def set_pressure_equation(self) -> None:

        advective_flux = self.mpfa.flux * self.p + self.mpfa.bound_flux * self.p_bc

        equation = self.div * advective_flux 
        
        equ_name = "pressure"
        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        self.ad_system.set_equation(equ_name, equation, num_equ_per_dof=image_info)
        self.system["primary_equations"].append(equ_name)

    def set_satur_equation(self) -> None:

        upwind_adv = self.upwind_sw
        upwind_adv_bc = self.upwind_sw_bc

        accumulation = self.mass_matrix.mass * self.rho_w * (
            self.sw - self.sw.previous_timestep()
        )

        advection_scalar = self.rho_w * self.sw
        advective_flux = self.mpfa.flux * self.p + self.mpfa.bound_flux * self.p_bc

        advection = (
            advective_flux * (upwind_adv.upwind * advection_scalar)
            - upwind_adv.bound_transport_dir * advective_flux * upwind_adv_bc
            - upwind_adv.bound_transport_neu * upwind_adv_bc
        )

        equation = accumulation + self.dt * (self.div * advection)
        
        equ_name = "mass_sw"
        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        self.ad_system.set_equation(equ_name, equation, num_equ_per_dof=image_info)
        self.system["primary_equations"].append(equ_name)

        if self._use_pressure_equation:
            return
        
        upwind_adv = self.upwind_sn
        upwind_adv_bc = self.upwind_sn_bc

        accumulation = self.mass_matrix.mass * self.rho_n * (
            self.sn - self.sn.previous_timestep()
        )

        advection_scalar = self.rho_n * self.sn
        advective_flux = self.mpfa.flux * self.p + self.mpfa.bound_flux * self.p_bc

        advection = (
            advective_flux * (upwind_adv.upwind * advection_scalar)
            - upwind_adv.bound_transport_dir * advective_flux * upwind_adv_bc
            - upwind_adv.bound_transport_neu * upwind_adv_bc
        )

        equation = accumulation + self.dt * (self.div * advection)

        equ_name = "mass_sn"
        image_info = dict()
        for sd in self.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        self.ad_system.set_equation(equ_name, equation, num_equ_per_dof=image_info)
        self.system["primary_equations"].append(equ_name)

    def _test(self):
        # div(A * F), where F is a flux and A a scalar to be upwinded
        F = (self.mpfa.flux * self.p).evaluate(self.dof_manager).val
        FBC = (self.mpfa.bound_flux * self.p_bc).evaluate(self.dof_manager)
        UP = self.upwind_sw.upwind.evaluate(self.dof_manager)
        UPDIR = self.upwind_sw.bound_transport_dir.evaluate(self.dof_manager)
        UPNEU = self.upwind_sw.bound_transport_neu.evaluate(self.dof_manager)
        UPBC = self.upwind_sw_bc.evaluate(self.dof_manager)

        A = (self.rho_w * self.sw).evaluate(self.dof_manager).val

        ADVF = F + FBC
        TOTAL = ADVF *(UP*A) - UPDIR * ADVF * UPBC
        FNEU = UPNEU * UPBC

        print(TOTAL)
        print(FNEU)
        print(TOTAL - FNEU)  # !!! NO OUTFLUX TODO

        return

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")
file_name = "testmodel_"  # + timestamp
params = {
    "folder_name": "/mnt/c/Users/vl-work/Desktop/sim-results/" + file_name + "/",
    "file_name": file_name,
    "use_ad": False,
    "use_pressure_equation": True,
    "monolithic": True,
}

t = 0.
T = 10.
dt = 0.1
max_iter = 200
tol = 1e-5

model =TestModel(params=params)

model.dt = dt
model.prepare_simulation()

while t < T:
    print(".. Timestep t=%f , dt=%e" % (t, model.dt))
    model.before_newton_loop()

    # if t > 3:
    #     model.test = True

    for i in range(1, max_iter + 1):
        model.before_newton_iteration()
        dx = model.assemble_and_solve_linear_system(tol)
        if model.converged:
            print(f"Success flow after iteration {i - 1}.")
            # model.print_x("convergence SUCCESS")
            model.after_newton_convergence(dx, tol, i - 1)
            break
        print(f".. .. flow iteration {i}.")
        model.after_newton_iteration(dx, i)

    if not model.converged:
        print(f"FAILURE: flow at time {t} after {max_iter} iterations.")
        # model.print_x("Flow convergence failure")
        model.after_newton_failure(dx, tol, max_iter)
        model.dt = model.dt / 2
        print(f"Halving timestep to {model.dt}")
        if model.dt < 0.001:
            model.after_simulation()
            raise RuntimeError("Time step halving due to convergence failure reached critical value.")
    else:
        t += model.dt
    
    if t >= T:
        print(f"Reached and of simulation: t={t}")

model.after_simulation()