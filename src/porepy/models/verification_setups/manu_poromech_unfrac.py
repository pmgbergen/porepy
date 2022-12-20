"""
This module contains an implementation of a verification setup for a poromechanical
system (without fractures) using a two-dimensional manufactured solution.

For the exact solution, we refer to https://doi.org/10.1137/15M1014280.

"""

import porepy as pp
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

import porepy.models.fluid_mass_balance as mass
import porepy.models.momentum_balance as momentum
import porepy.models.poromechanics as poromechanics

from porepy.models.verification_setups.verifications_utils import VerificationUtils

from typing import Callable


# ----> Manufactured solution
class ExactSolution:
    """Parent class for the manufactured poromechanical solution."""

    def __init__(self):
        """Constructor of the class."""

        # FIXME: Retrieve physical parameters from the `setup`
        # Physical parameters
        lame_lmbda = 1.0  # [Pa] Lamé parameter
        lame_mu = 1.0  # [Pa] Lamé parameter
        alpha = 0.8  # [-] Biot's coefficient
        rho_0 = 1.0  # [kg * m^-3] Reference fluid density
        phi_0 = 0.10  # [-] Reference porosity
        p_0 = 0.0  # [Pa] Reference fluid pressure
        c_f = 0.15  # [Pa^-1] Fluid compressibility
        k = 1.0  # [m^2] Permeability
        mu_f = 1.0  # [Pa * s] Dynamic fluid viscosity
        K_d = lame_lmbda + (2 / 3) * lame_mu  # [Pa] Bulk modulus TODO: Is this right?

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Exact pressure
        p = t * x * (1 - x) * sym.sin(2 * sym.pi * y)

        # Exact displacement
        u = [
            t * x * (1 - x) * sym.sin(2 * sym.pi * y),
            t * sym.sin(2 * sym.pi * x) * sym.sin(2 * sym.pi * y),
        ]

        # Exact density
        rho = rho_0 * sym.exp(c_f * (p - p_0))

        # Exact darcy flux
        q = [-(k / mu_f) * sym.diff(p, x), -(k / mu_f) * sym.diff(p, y)]

        # Exact mass flux
        mf = [rho * q[0], rho * q[1]]

        # Exact divergence of the mass flux
        div_mf = sym.diff(mf[0], x) + sym.diff(mf[1], y)

        # Exact divergence of the displacement
        div_u = sym.diff(u[0], x) + sym.diff(u[1], y)

        # Exact poromechanical porosity
        phi = phi_0 + ((alpha - phi_0) * (1 - alpha) / K_d) * (p - p_0) + alpha * div_u

        # Exact flow accumulation
        accum_flow = sym.diff(phi * rho, t)

        # Exact flow source
        source_flow = accum_flow + div_mf

        # Exact gradient of the displacement
        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y)],
            [sym.diff(u[1], x), sym.diff(u[1], y)],
        ]

        # Exact transpose of the gradient of the displacement
        trans_grad_u = [[grad_u[0][0], grad_u[1][0]], [grad_u[0][1], grad_u[1][1]]]

        # Exact strain
        epsilon = [
            [
                0.5 * (grad_u[0][0] + trans_grad_u[0][0]),
                0.5 * (grad_u[0][1] + trans_grad_u[0][1]),
            ],
            [
                0.5 * (grad_u[1][0] + trans_grad_u[1][0]),
                0.5 * (grad_u[1][1] + trans_grad_u[1][1]),
            ],
        ]

        # Exact trace (in the linear algebra sense) of the strain
        tr_epsilon = epsilon[0][0] + epsilon[1][1]

        # Exact elastic stress
        sigma_elas = [
            [
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[0][0],
                2 * lame_mu * epsilon[0][1],
            ],
            [
                2 * lame_mu * epsilon[1][0],
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[1][1],
            ],
        ]

        # Exact poroelastic stress
        sigma_total = [
            [sigma_elas[0][0] - alpha * p, sigma_elas[0][1]],
            [sigma_elas[1][0], sigma_elas[1][1] - alpha * p],
        ]

        # Mechanics source term
        source_mech = [
            sym.diff(sigma_total[0][0], x) + sym.diff(sigma_total[1][0], y),
            sym.diff(sigma_total[0][1], x) + sym.diff(sigma_total[1][1], y),
        ]

        # Public attributes
        self.p = p  # pressure
        self.u = u  # displacement
        self.sigma_elast = sigma_elas  # elastic stress
        self.sigma_total = sigma_total  # poroelastic (total) stress
        self.q = q  # Darcy flux
        self.mf = mf  # Mass flux, i.e., density * Darcy flux
        self.phi = phi  # poromechanics porosity
        self.rho = rho  # density
        self.source_mech = source_mech  # Source term entering the momentum balance
        self.source_flow = source_flow  # Source term entering the mass balance

    def pressure(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact pressure [Pa] at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact pressures at the
            cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        p_fun: Callable = sym.lambdify((x, y, t), self.p, "numpy")

        # Cell-centered pressures
        p_cc: np.ndarray = p_fun(cc[0], cc[1], time)

        return p_cc

    def displacement(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact displacement [m] at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_cells, )`` containing the exact displacements
            at the cell centers for the given ``time``.

        Notes:
            The returned displacement is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        u_fun: list[Callable] = [
            sym.lambdify((x, y, t), self.u[0], "numpy"),
            sym.lambdify((x, y, t), self.u[1], "numpy"),
        ]

        # Cell-centered displacements
        u_cc: list[np.ndarray] = [
            u_fun[0](cc[0], cc[1], time),
            u_fun[1](cc[0], cc[1], time),
        ]

        # Flatten array
        u_flat: np.ndarray = np.ravel(np.array(u_cc), "F")

        return u_flat

    def elastic_force(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact elastic force [N] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact elastic force
            at the face centers for the given ``time``.

        Notes:
            - The returned elastic force is given in PorePy's flattened vector format.
            - Recall that force = (stress \dot unit_normal) * face_area.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        sigma_elas_fun: list[list[Callable]] = [
            [
                sym.lambdify((x, y, t), self.sigma_elast[0][0], "numpy"),
                sym.lambdify((x, y, t), self.sigma_elast[0][1], "numpy"),
            ],
            [
                sym.lambdify((x, y, t), self.sigma_elast[1][0], "numpy"),
                sym.lambdify((x, y, t), self.sigma_elast[1][1], "numpy"),
            ],
        ]

        # Face-centered elastic force
        force_elast_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_elas_fun[0][0](fc[0], fc[1], time) * fn[0]
            + sigma_elas_fun[0][1](fc[0], fc[1], time) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_elas_fun[1][0](fc[0], fc[1], time) * fn[0]
            + sigma_elas_fun[1][1](fc[0], fc[1], time) * fn[1],
        ]

        # Flatten array
        force_elast_flat: np.ndarray = np.asarray(force_elast_fc).ravel("F")

        return force_elast_flat

    def poroelastic_force(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact poroelastic force [N] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact poroealstic
            force at the face centers for the given ``time``.

        Notes:
            - The returned poroelastic force is given in PorePy's flattened vector
              format.
            - Recall that force = (stress \dot unit_normal) * face_area.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        sigma_total_fun: list[list[Callable]] = [
            [
                sym.lambdify((x, y, t), self.sigma_total[0][0], "numpy"),
                sym.lambdify((x, y, t), self.sigma_total[0][1], "numpy"),
            ],
            [
                sym.lambdify((x, y, t), self.sigma_total[1][0], "numpy"),
                sym.lambdify((x, y, t), self.sigma_total[1][1], "numpy"),
            ],
        ]

        # Face-centered poroelastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[0][1](fc[0], fc[1], time) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[1][1](fc[0], fc[1], time) * fn[1],
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")

        return force_total_flat

    def darcy_flux(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact Darcy flux [m^3 * s^-1] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact Darcy fluxes at
            the face centers for the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd.face_normals``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        q_fun: list[Callable, Callable] = [
            sym.lambdify((x, y, t), self.q[0], "numpy"),
            sym.lambdify((x, y, t), self.q[1], "numpy"),
        ]

        # Face-centered Darcy fluxes
        q_fc: np.ndarray = (
            q_fun[0](fc[0], fc[1], time) * fn[0] + q_fun[1](fc[0], fc[1], time) * fn[1]
        )

        return q_fc

    def mass_flux(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact mass flux [kg * s^-1] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact mass fluxes at
            the face centers for the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        mf_fun: list[Callable, Callable] = [
            sym.lambdify((x, y, t), self.mf[0], "numpy"),
            sym.lambdify((x, y, t), self.mf[1], "numpy"),
        ]

        # Face-centered mass fluxes
        mf_fc: np.ndarray = (
            mf_fun[0](fc[0], fc[1], time) * fn[0]
            + mf_fun[1](fc[0], fc[1], time) * fn[1]
        )

        return mf_fc

    def mechanics_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the momentum balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the momentum balance equation with ``shape=(
            2 * sd.num_cells, )``.

        Notes:
            The returned array is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers and cell volumes
        cc = sd.cell_centers
        vol = sd.cell_volumes

        # Lambdify expression
        source_mech_fun: list[Callable] = [
            sym.lambdify((x, y, t), self.source_mech[0], "numpy"),
            sym.lambdify((x, y, t), self.source_mech[1], "numpy"),
        ]

        # Evaluate and integrate source
        source_mech: list[np.ndarray] = [
            source_mech_fun[0](cc[0], cc[1], time) * vol,
            source_mech_fun[1](cc[0], cc[1], time) * vol,
        ]

        # Flatten array
        source_mech_flat: np.ndarray = np.asarray(source_mech).ravel("F")

        return source_mech_flat

    def flow_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the fluid mass balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the fluid mass balance equation with ``shape=(
            sd.num_cells, )``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers and cell volumes
        cc = sd.cell_centers
        vol = sd.cell_volumes

        # Lambdify expression
        source_flow_fun: Callable = sym.lambdify((x, y, t), self.source_flow, "numpy")

        # Evaluate and integrate source
        source_flow: np.ndarray = source_flow_fun(cc[0], cc[1], time) * vol

        return source_flow


# ----> Results storage
class StoreResults(VerificationUtils):
    """Class for storing results."""

    def __init__(self, setup):
        """Constructor of the class"""

        # Retrieve information from setup
        mdg: pp.MixedDimensionalGrid = setup.mdg
        sd: pp.Grid = mdg.subdomains()[0]
        data: dict = mdg.subdomain_data(sd)
        t: float = setup.time_manager.time
        ex: ExactSolution = setup.exact_sol
        p_name: str = setup.pressure_variable
        q_name: str = "darcy_flux"
        u_name: str = setup.displacement_variable
        force_name: str = "poroelastic_force"

        # Store time for convenience
        self.time = t

        # Pressure
        self.exact_pressure = ex.pressure(sd=sd, time=t)
        self.approx_pressure = data[pp.STATE][pp.ITERATE][p_name]
        self.error_pressure = self.relative_l2_error(
            grid=sd,
            true_array=self.exact_pressure,
            approx_array=self.approx_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Darcy flux
        self.exact_flux = ex.darcy_flux(sd=sd, time=t)
        self.approx_flux = data[pp.STATE][pp.ITERATE][q_name]
        self.error_flux = self.relative_l2_error(
            grid=sd,
            true_array=self.exact_flux,
            approx_array=self.approx_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Displacement
        self.exact_displacement = ex.displacement(sd=sd, time=t)
        self.approx_displacement = data[pp.STATE][pp.ITERATE][u_name]
        self.error_displacement = self.relative_l2_error(
            grid=sd,
            true_array=self.exact_displacement,
            approx_array=self.approx_displacement,
            is_scalar=False,
            is_cc=True,
        )

        # Poroelastic force
        self.exact_force = ex.poroelastic_force(sd=sd, time=t)
        self.approx_force = data[pp.STATE][pp.ITERATE][force_name]
        self.error_displacement = self.relative_l2_error(
            grid=sd,
            true_array=self.exact_force,
            approx_array=self.approx_force,
            is_scalar=False,
            is_cc=False,
        )


# ----> Simulation model

# --------> Geometry
class UnitSquare(pp.ModelGeometry):
    def set_fracture_network(self) -> None:
        domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        self.fracture_network = pp.FractureNetwork2d(None, None, domain)

    def mesh_arguments(self) -> dict:
        default_mesh_arguments = {"mesh_size_frac": 0.05, "mesh_size_bound": 0.05}
        return self.params.get("mesh_arguments", default_mesh_arguments)

    def set_md_grid(self) -> None:
        self.mdg = self.fracture_network.mesh(self.mesh_arguments())
        self.domain_bounds = self.fracture_network.domain


# --------> Equations
class ModifiedMassBalance(mass.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid source."""

        # Internal sources are inherit from parent class
        # Zero internal sources for unfractured domains, but we keep it to maintain
        # good code practice
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # External sources are retrieved from STATE and wrapped as an AdArray
        external_sources = pp.ad.TimeDependentArray(
            name="source_flow",
            subdomains=self.mdg.subdomains(),
            previous_timestep=True,
        )

        return internal_sources + external_sources


class ModifiedMomentumBalance(momentum.MomentumBalanceEquations):
    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Mechanics source term."""

        external_sources = pp.ad.TimeDependentArray(
            name="source_mechanics",
            subdomains=self.mdg.subdomains(),
            previous_timestep=True,
        )

        return external_sources


class ModifiedEquationsPoromechanics(ModifiedMassBalance, ModifiedMomentumBalance):
    ...


# ---------> Solution Strategy
class ModifiedSolutionStrategy(poromechanics.SolutionStrategyPoromechanics):
    """Solution strategy for the verification setup."""

    fluid: pp.FluidConstants
    mdg: pp.MixedDimensionalGrid
    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    stress: Callable[[list[pp.Grid]], pp.ad.Operator]
    results: list[StoreResults]

    def __init__(self, params: dict):
        """Constructor for the class"""

        # Parameters associated with the verification setup. The below parameters
        # cannot be changed since they're associated with the exact solution
        fluid = pp.FluidConstants({"compressibility": 0.15})
        solid = pp.SolidConstants({"porosity": 0.10, "biot_coefficient": 0.8})
        material_constants = {"fluid": fluid, "solid": solid}
        required_params = {"material_constants": material_constants}
        params.update(required_params)

        # Default time manager
        default_time_manager = pp.TimeManager(
            schedule=[0, 0.2, 0.4, 0.6, 0.8, 1.0], dt_init=0.2, constant_dt=True
        )
        params.update({"time_manager": default_time_manager})

        super().__init__(params)

        # Exact solution
        self.exact_sol = ExactSolution()
        """Exact solution object"""

        # Prepare object to store solutions after each time step
        self.results: list[StoreResults] = []
        """Object that stores exact and approximated solutions and L2 errors"""

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""

        # Retrieve subdomain and data dictionary
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)

        # Retrieve current time
        t = self.time_manager.time

        # Mechanics source
        mech_source = self.exact_sol.mechanics_source(sd=sd, time=t)
        data[pp.STATE]["source_mechanics"] = mech_source

        # Flow source
        flow_source = self.exact_sol.flow_source(sd=sd, time=t)
        data[pp.STATE]["source_flow"] = flow_source

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after convergence."""

        # Subomain grid and data dictionary
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)

        # Recover Darcy fluxes and store in pp.ITERATE
        darcy_flux_ad = self.darcy_flux([sd])
        darcy_flux_vals = darcy_flux_ad.evaluate(self.equation_system).val
        data[pp.STATE][pp.ITERATE]["darcy_flux"] = darcy_flux_vals

        # Recover poroelastic force and store in pp.ITERATE
        force_ad = self.stress([sd])
        force_vals = force_ad.evaluate(self.equation_system).val
        data[pp.STATE][pp.ITERATE]["poroelastic_force"] = force_vals

        # Store results
        t = self.time_manager.time
        if np.any(np.isclose(t, self.params["stored_times"])):
            self.results.append(StoreResults(self))

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""

        if self.params.get("plot_results", False):
            self.plot_results()

    # -----> Plotiing-related methods
    def plot_results(self) -> None:
        """Plotting results"""

        self._plot_displacement()
        self._plot_pressure()

    def _plot_displacement(self):
        sd = self.mdg.subdomains()[0]
        u_ex = self.results[-1].exact_displacement
        u_num = self.results[-1].approx_displacement

        # Horizontal displacement
        pp.plot_grid(sd, u_ex[::2], plot_2d=True, linewidth=0, title="u_x (Exact)")
        pp.plot_grid(sd, u_num[::2], plot_2d=True, linewidth=0, title="u_x (MPFA)")

        # Vertical displacement
        pp.plot_grid(sd, u_ex[1::2], plot_2d=True, linewidth=0, title="u_y (Exact)")
        pp.plot_grid(sd, u_num[1::2], plot_2d=True, linewidth=0, title="u_y (MPFA)")

    def _plot_pressure(self):
        sd = self.mdg.subdomains()[0]
        p_ex = self.results[-1].exact_pressure
        p_num = self.results[-1].approx_pressure

        pp.plot_grid(sd, p_ex, plot_2d=True, linewidth=0, title="p (Exact)")
        pp.plot_grid(sd, p_num, plot_2d=True, linewidth=0, title="p (MPFA)")


# ---------> Mixer class
class ManuPoromechanics2d(
    UnitSquare,
    ModifiedEquationsPoromechanics,
    ModifiedSolutionStrategy,
    poromechanics.Poromechanics,
):
    """
    Mixer class for the verification setup.

    Examples:

        .. code:: python

            # Import modules
            import porepy as pp
            from time import time

            # Run verification setup
            tic = time()
            params = {"plot_results": True}
            setup = ManuPoromechanics2d(params)
            print("Simulation started...")
            pp.run_time_dependent_model(setup, params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds.")

    """

    ...


#%% Runner
from time import time

tic = time()
params = {
    "plot_results": True,
    "stored_times": [1.0],
    "mesh_arguments": {"mesh_size_frac": 0.05, "mesh_size_bound": 0.05},
}
setup = ManuPoromechanics2d(params)
print("Simulation started...")
pp.run_time_dependent_model(setup, params)
toc = time()
print(f"Simulation finished in {round(toc - tic)} seconds.")
