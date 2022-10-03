import sys
sys.path.append("/mnt/c/Users/vl-work/Desktop/github/porepy/src")

import numpy as np
import porepy as pp
import scipy.sparse as sps

from porepy.models.abstract_model import AbstractModel


class ExampleFlow(AbstractModel):
    """
    
    d_t rho + div grad p = 0
    rho = rho_0 * (1 + c_0 * (p - p_0))
    
    """

    def __init__(self, params = None):
        super().__init__(params)

        # declare data structures
        self.exporter: pp.Exporter
        self.mdg: pp.MixedDimensionalGrid
        self.box: dict
        self.dof_man: pp.DofManager
        self.ad_sys: pp.ad.ADSystemManager
        self.p: pp.ad.MergedVariable
        self.rho: pp.ad.MergedVariable
        self.div: pp.ad.Operator
        self.linear_system: tuple[sps.spmatrix, np.ndarray]
        self.mpfa: pp.ad.MpfaAd
        self.bc: pp.ad.BoundaryCondition

        # declare variables
        self.primary_vars = {
            "pressure": {"cells": 1},
        }
        self.secondary_vars = {
            "density": {"cells": 1},
        }
        self.mortar_vars = {
            "mortar_pressure" : {"cells": 1},
        }

        # declare secondary equations
        self.primary_equations = list()

        # model paramethers
        self.rho0 = 1.
        self.c0 = 1.
        self.p0 = 1.
        self.dt = 0.1
        self.inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

        self.flow_parameter_key = "flow"
        self.energy_parameter_key = "heat"
    
    def prepare_simulation(self) -> None:
        # prepare data structures
        self.create_grid()
        self.dof_man = pp.DofManager(self.mdg)
        self.ad_sys = pp.ad.ADSystemManager(self.dof_man)

        self.exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
        )

        # creating AD variables
        self._set_variables()

        # setting parameters for grid and numerical approach
        self._set_parameters()

        self.mpfa = pp.ad.MpfaAd(self.flow_parameter_key, self.mdg.subdomains())
        self.bc = pp.ad.BoundaryCondition(self.flow_parameter_key, self.mdg.subdomains())

        # setting equations
        self._set_model_equations()

        # discretizing
        self.ad_sys.discretize()
    
    def create_grid(self):
        phys_dims = [1, 1]
        n_cells = [10,10]
        bounding_box_points = np.array([[0, phys_dims[0]],[0, phys_dims[1]]])
        self.box = pp.geometry.bounding_box.from_points(bounding_box_points)
        f1 = np.array([[0.3, 0.5],[0.7, 0.5]]).T
        self.mdg = pp.meshing.cart_grid(
            fracs = [f1], physdims = phys_dims, nx = np.array(n_cells)
        )

    def _set_variables(self):
        # primary variable
        self.p = self.ad_sys.create_variable("pressure", True, self.primary_vars["pressure"])
        # secondary variable
        self.rho = self.ad_sys.create_variable(
            "density", False, self.secondary_vars["density"]
        )
        # mortar variable
        infts = self.mdg.interfaces()
        self.mortar_p = self.ad_sys.create_variable(
            "mortar_pressure", True, self.mortar_vars["mortar_pressure"], subdomains=None, interfaces=infts
        )

        # setting initial values
        vals = np.ones(self.mdg.num_subdomain_cells())
        self.ad_sys.set_var_values("pressure", np.copy(self.p0 * vals), True)
        self.ad_sys.set_var_values("density", np.copy(self.rho0 * vals), True)

        vals = np.ones(self.mdg.num_interface_cells())
        self.ad_sys.set_var_values("mortar_pressure", np.copy(self.p0 * vals), True)
    
    def _set_parameters(self):
        for g, d in self.mdg.subdomains(return_data=True):
            unit_tensor = pp.SecondOrderTensor(np.ones(g.num_cells))
            zero_vector_source = np.zeros((self.mdg.dim_max(), g.num_cells))
            all_idx, *_= self._domain_boundary_sides(g)
            bc = pp.BoundaryCondition(g, all_idx, "dir")
            vals = np.zeros(g.num_faces)
            vals[all_idx] = 1.0
            # parameters for MPFA discretization
            pp.initialize_data(
                g,
                d,
                self.flow_parameter_key,
                {
                    "bc": bc,
                    "bc_values": vals,
                    "second_order_tensor": unit_tensor,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                    "mass_weight": np.ones(g.num_cells),
                },
            )

            bc = pp.BoundaryCondition(g, all_idx, "dir")
            vals = np.zeros(g.num_faces)
            vals[all_idx] = 10.0

            pp.initialize_data(
                g,
                d,
                self.energy_parameter_key,
                {
                    "bc": bc,
                    "bc_values": vals,
                    "second_order_tensor": unit_tensor,
                    "vector_source": np.copy(zero_vector_source.ravel("F")),
                    "ambient_dimension": self.mdg.dim_max(),
                },
            )

    def _set_model_equations(self):
        
        ### primary equation
        # accumulation
        equ = self.rho - self.rho.previous_timestep()

        self.div = pp.ad.Divergence(subdomains=self.mdg.subdomains())

        flux = self.mpfa.flux * self.p + self.mpfa.bound_flux * self.bc

        equ += self.dt * self.div * flux

        self.ad_sys.set_equation("mass_balance", equ)
        # mark as primary equation
        self.primary_equations.append("mass_balance")

        ### interface equation pressure (exponential decay)
        equ = (
            self.mortar_p - self.mortar_p.previous_timestep()
            + self.dt * self.dt / (2 * np.pi) * self.mortar_p
        )
        self.ad_sys.set_equation("interface_pressure", equ)
        self.primary_equations.append("interface_pressure")

        ### secondary equation
        equ = self.rho - self.rho0 *(1 + self.c0 * (self.p - self.p0))
        self.ad_sys.set_equation("density_law", equ)

    def before_newton_loop(self):
        self._nonlinear_iteration = 0

    def before_newton_iteration(self):
        pass

    def assemble_linear_system(self) -> None:
        A, b = self.ad_sys.assemble_schur_complement_system(
            self.primary_equations,
            list(self.primary_vars.keys()) + list(self.mortar_vars.keys()),
            self.inverter
        )
        self.linear_system = (A, b)

    def after_newton_iteration(self, solution_vector):
        self._nonlinear_iteration += 1
        solution_vector = self.ad_sys.expand_schur_complement_solution(solution_vector)
        self.dof_man.distribute_variable(
            values=solution_vector, additive=True, to_iterate=True
        )

    def after_newton_convergence(self, solution, errors, iteration_counter):
        solution = self.dof_man.assemble_variable(from_iterate=True)
        self.dof_man.distribute_variable(values=solution)
        self.convergence_status = True
        self._export()

    def after_simulation(self):
        pass

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu(
                list(self.primary_vars.keys()) + list(self.secondary_vars.keys())
            )
        
    def _is_nonlinear_problem(self) -> bool:
        False



class ExampleExtension(ExampleFlow):
    """Appends the energy equation with sources
    
    d_t rho*c*T - div(rho*c*T* grad p) - div( k * grad T ) = q_T

    """

    def __init__(self, params=None):
        super().__init__(params)

        self.T: pp.ad.MergedVariable
        self.mortar_T: pp.ad.MergedVariable

        self.primary_vars.update({
            "temperature": {"cells": 1},
        })

        self.mortar_vars.update({
            "mortar_temperature": {"cells": 1},
        })

        self.heat_cap = 1.
        self.conduct = 1.
        self.T0 = 1.
        self.thermal_exp = 1.

    def _set_variables(self):
        super()._set_variables()

        # adding new primary
        self.T = self.ad_sys.create_variable(
            "temperature", True, self.primary_vars["temperature"]
        )
        # adding new mortar var
        intfs = self.mdg.interfaces()
        self.mortar_T = self.ad_sys.create_variable(
            "mortar_temperature", True,
            self.mortar_vars["mortar_temperature"],subdomains=None, interfaces=intfs
            )
        
        # initial values
        vals = np.ones(self.mdg.num_subdomain_cells())
        self.ad_sys.set_var_values("temperature", self.T0 * vals, True)
        
        vals = np.ones(self.mdg.num_interface_cells())
        self.ad_sys.set_var_values("mortar_temperature", self.T0 * vals, True)
    
    def _set_model_equations(self):
        super()._set_model_equations()

        # adjusting the secondary equation with thermal expansion
        equ = self.ad_sys.equations["density_law"]

        equ -= self.rho0 * self.thermal_exp * (self.T - self.T0)

        # adding a new primary equation
        # accumulation
        equ = (
            self.heat_cap * self.T * (self.rho - self.rho.previous_timestep())
            + self.rho * self.heat_cap * (self.T - self.T.previous_timestep())
        )

        # conductive flux
        conductive_heat_flux = self.mpfa.flux * self.T + self.mpfa.bound_flux * self.bc

        equ -= self.dt * self.div * (self.conduct * conductive_heat_flux)

        # convective flux
        # convective_heat_flux: pp.ad.Operator
        # equ -= self.dt * self.div * (convective_heat_flux)

        heat_source = pp.ad.Scalar(1.)

        equ -= heat_source

        self.ad_sys.set_equation("energy_balance", equ)
        # mark as primary equation
        self.primary_equations.append("energy_balance")

        # adding interface equation for heat flux (exponential decay)
        equ = (
            self.mortar_T - self.mortar_T.previous_timestep()
            + self.dt * self.dt / (2 * np.pi) * self.mortar_T
        )
        self.ad_sys.set_equation("interface_heat", equ)
        self.primary_equations.append("interface_heat")
    
    def _is_nonlinear_problem(self) -> bool:
        True


M = ExampleExtension()
M.prepare_simulation()

M.assemble_linear_system()
A, b = M.linear_system

dx_s = sps.linalg.spsolve(A,b)

M.after_newton_iteration(dx_s)