"""
This module contains the implementation of a verification setup for a poromechanical
system (without fractures) using two-dimensional manufactured solutions.

The implementation considers a compressible fluid, resulting in a coupled non-linear
set of equations. The dependecy of the fluid density with the pressure is given by [1]:

.. math:

    \\rho(p) =  \\rho_0 \\exp{c_f \\left(p - p_0\\right)},

where :math:`\\rho` and :math:`p` are the density and pressure, :math:`\\rho_0`` and
:math:``p_0`` are the _reference_ density and pressure, and :math:`c_f` is the
_constant_ fluid compressibility.

We provide three different manufactured solutions, namely `parabolic`,
`nordbotten2016` [2], and `varela2021` [2], which can be specified as a model
parameter.

Parabolic manufactured solution:

.. math:

    p(x, y, t) = t x (1 - x) y (1- y),

    u_x(x, y, t) = p(x, y, t),

    u_y(x, y, t) = p(x, y, t).

Manufactured solution based on [2]:

.. math:

    p(x, y, t) = t x (1 - x) \\sin{2 \\pi y},

    u_x(x, y, t) = p(x, y, t),

    u_y(x, y, t) = t * \\sin{2 \\pi x} \\sin{2 \\pi y}.

Manufactured solutions based on [3]:

.. math:

    p(x, y, t) = t x (1 - x) y (1 - y) \\sin{\\pi x} \\sin{\\pi y},

    u(x, y, t) = t x (1 - x) y (1 - y) \\sin{\\pi x},

    u(x, y, t) = t x (1 - x) y (1 - y) \\cos{\\pi x}.

References:

    - [1] Garipov, T. T., & Hui, M. H. (2019). Discrete fracture modeling approach for
      simulating coupled thermo-hydro-mechanical effects in fractured reservoirs.
      International Journal of Rock Mechanics and Mining Sciences, 122, 104075.

    - [2] Nordbotten, J. M. (2016). Stable cell-centered finite volume discretization
      for Biot equations. SIAM Journal on Numerical Analysis, 54(2), 942-968.

    - [3] Varela, J., Gasda, S. E., Keilegavlen, E., & Nordbotten, J. M. (2021). A
      Finite-Volume-Based Module for Unsaturated Poroelasticity. Advanced Modeling
      with the MATLAB Reservoir Simulation Toolbox.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sym

import porepy as pp
import porepy.models.fluid_mass_balance as mass
import porepy.models.momentum_balance as momentum
import porepy.models.poromechanics as poromechanics
from porepy.applications.building_blocks.verification_utils import VerificationUtils
from porepy.viz.data_saving_model_mixin import VerificationDataSaving

# PorePy typings
number = pp.number
grid = pp.GridLike


# -----> Data-saving
@dataclass
class ManuPoroMechSaveData:
    """Data class to save relevant results from the verification setup."""

    approx_displacement: np.ndarray
    """Numerical displacement."""

    approx_flux: np.ndarray
    """Numerical Darcy flux."""

    approx_force: np.ndarray
    """Numerical poroelastic force."""

    approx_pressure: np.ndarray
    """Numerical pressure."""

    error_displacement: number
    """L2-discrete relative error for the displacement."""

    error_flux: number
    """L2-discrete relative error for the Darcy flux."""

    error_force: number
    """L2-discrete relative error for the poroelastic force."""

    error_pressure: number
    """L2-discrete relative error for the pressure."""

    exact_displacement: np.ndarray
    """Exact displacement."""

    exact_flux: np.ndarray
    """Exact Darcy flux."""

    exact_force: np.ndarray
    """Exact poroelastic force."""

    exact_pressure: np.ndarray
    """Exact pressure."""

    time: number
    """Current simulation time."""


class ManuPoroMechDataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`porepy.models.constitutive_laws.DarcysLaw`.

    """

    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """

    exact_sol: ManuPoroMechExactSolution
    """Exact solution object."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    stress: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the (integrated) poroelastic stress in the form of an Ad
    operator. Usually provided by the mixin class
    :class:`porepy.models.poromechanics.ConstitutiveLawsPoromechanics`.

    """

    relative_l2_error: Callable
    """Method for computing the discrete relative L2-error. Normally provided by a
    mixin instance of :class:`~porepy.applications.building_blocks.
    verification_utils.VerificationUtils`.

    """

    def collect_data(self) -> ManuPoroMechSaveData:
        """Collect data from the verification setup.

        Returns:
            ManuPoroMechSaveData object containing the results of the verification for
            the current time.

        """

        mdg: pp.MixedDimensionalGrid = self.mdg
        sd: pp.Grid = mdg.subdomains()[0]
        t: number = self.time_manager.time

        # Collect data
        exact_pressure = self.exact_sol.pressure(sd=sd, time=t)
        pressure_ad = self.pressure([sd])
        approx_pressure = pressure_ad.evaluate(self.equation_system).val
        error_pressure = self.relative_l2_error(
            grid=sd,
            true_array=exact_pressure,
            approx_array=approx_pressure,
            is_scalar=True,
            is_cc=True,
        )

        exact_displacement = self.exact_sol.displacement(sd=sd, time=t)
        displacement_ad = self.displacement([sd])
        approx_displacement = displacement_ad.evaluate(self.equation_system).val
        error_displacement = self.relative_l2_error(
            grid=sd,
            true_array=exact_displacement,
            approx_array=approx_displacement,
            is_scalar=False,
            is_cc=True,
        )

        exact_flux = self.exact_sol.darcy_flux(sd=sd, time=t)
        flux_ad = self.darcy_flux([sd])
        approx_flux = flux_ad.evaluate(self.equation_system).val
        error_flux = self.relative_l2_error(
            grid=sd,
            true_array=exact_flux,
            approx_array=approx_flux,
            is_scalar=True,
            is_cc=False,
        )

        exact_force = self.exact_sol.poroelastic_force(sd=sd, time=t)
        force_ad = self.stress([sd])
        approx_force = force_ad.evaluate(self.equation_system).val
        error_force = self.relative_l2_error(
            grid=sd,
            true_array=exact_force,
            approx_array=approx_force,
            is_scalar=False,
            is_cc=False,
        )

        # Store collected data in data class
        collected_data = ManuPoroMechSaveData(
            approx_displacement=approx_displacement,
            approx_flux=approx_flux,
            approx_force=approx_force,
            approx_pressure=approx_pressure,
            error_displacement=error_displacement,
            error_flux=error_flux,
            error_force=error_force,
            error_pressure=error_pressure,
            exact_displacement=exact_displacement,
            exact_flux=exact_flux,
            exact_force=exact_force,
            exact_pressure=exact_pressure,
            time=t,
        )

        return collected_data


# -----> Exact solution
class ManuPoroMechExactSolution:
    """Class containing the exact manufactured solution for the verification setup."""

    def __init__(self, setup):
        """Constructor of the class."""

        # Physical parameters
        lame_lmbda = setup.solid.lame_lambda()  # [Pa] Lamé parameter
        lame_mu = setup.solid.shear_modulus()  # [Pa] Lamé parameter
        alpha = setup.solid.biot_coefficient()  # [-] Biot coefficient
        rho_0 = setup.fluid.density()  # [kg / m^3] Reference density
        phi_0 = setup.solid.porosity()  # [-] Reference porosity
        p_0 = setup.fluid.pressure()  # [Pa] Reference pressure
        c_f = setup.fluid.compressibility()  # [Pa^-1] Fluid compressibility
        k = setup.solid.permeability()  # [m^2] Permeability
        mu_f = setup.fluid.viscosity()  # [Pa * s] Fluid viscosity
        K_d = lame_lmbda + (2 / 3) * lame_mu  # [Pa] Bulk modulus

        # Symbolic variables
        x, y, t = sym.symbols("x y t")
        pi = sym.pi

        # Exact pressure and displacement solutions
        manufactured_sol = setup.params.get("manufactured_solution", "parabolic")
        if manufactured_sol == "parabolic":
            p = t * x * (1 - x) * y * (1 - y)
            u = [p, p]
        elif manufactured_sol == "nordbotten_2016":
            p = t * x * (1 - x) * sym.sin(2 * pi * y)
            u = [p, t * sym.sin(2 * pi * x) * sym.sin(2 * pi * y)]
        elif manufactured_sol == "varela_2021":
            p = t * x * (1 - x) * y * (1 - y) * sym.sin(pi * x) * sym.sin(pi * y)
            u = [
                t * x * (1 - x) * y * (1 - y) * sym.sin(pi * x),
                t * x * (1 - x) * y * (1 - y) * sym.cos(pi * x),
            ]
        else:
            NotImplementedError("Manufactured solution is not available.")

        # Exact density
        rho = rho_0 * sym.exp(c_f * (p - p_0))

        # Exact darcy flux
        q = [-1 * (k / mu_f) * sym.diff(p, x), -1 * (k / mu_f) * sym.diff(p, y)]

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

    # -----> Primary and secondary variables
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
        u_flat: np.ndarray = np.asarray(u_cc).ravel("F")

        return u_flat

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

        q_fun: list[Callable] = [
            sym.lambdify((x, y, t), self.q[0], "numpy"),
            sym.lambdify((x, y, t), self.q[1], "numpy"),
        ]

        # Face-centered Darcy fluxes
        q_fc: np.ndarray = (
            q_fun[0](fc[0], fc[1], time) * fn[0] + q_fun[1](fc[0], fc[1], time) * fn[1]
        )

        return q_fc

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
            - Recall that force = (stress dot_prod unit_normal) * face_area.

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

    # -----> Sources
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


# -----> Utilities
class ManuPoroMechUtils(VerificationUtils):
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[ManuPoroMechSaveData]
    """List of ManuPoroMechSaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_displacement()
        self._plot_pressure()

    def _plot_pressure(self):
        """Plot exact and numerical pressures."""

        sd = self.mdg.subdomains()[0]
        p_ex = self.results[-1].exact_pressure
        p_num = self.results[-1].approx_pressure

        pp.plot_grid(sd, p_ex, plot_2d=True, linewidth=0, title="p (Exact)")
        pp.plot_grid(sd, p_num, plot_2d=True, linewidth=0, title="p (Numerical)")

    def _plot_displacement(self):
        """Plot exact and numerical displacements."""

        sd = self.mdg.subdomains()[0]
        u_ex = self.results[-1].exact_displacement
        u_num = self.results[-1].approx_displacement

        # Horizontal displacement
        pp.plot_grid(sd, u_ex[::2], plot_2d=True, linewidth=0, title="u_x (Exact)")
        pp.plot_grid(sd, u_num[::2], plot_2d=True, linewidth=0, title="u_x (Numerical)")

        # Vertical displacement
        pp.plot_grid(sd, u_ex[1::2], plot_2d=True, linewidth=0, title="u_y (Exact)")
        pp.plot_grid(
            sd, u_num[1::2], plot_2d=True, linewidth=0, title="u_y (Numerical)"
        )


# -----> Geometry
class UnitSquareTriangleGrid(pp.ModelGeometry):
    """Class for setting up the geometry of the unit square domain."""

    params: dict
    """Simulation model parameters."""

    fracture_network: pp.FractureNetwork2d
    """Fracture network. Empty in this case."""

    def set_fracture_network(self) -> None:
        """Set fracture network. Unit square with no fractures."""
        domain = pp.Domain({"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0})
        self.fracture_network = pp.FractureNetwork2d(domain=domain)

    def mesh_arguments(self) -> dict:
        """Set mesh arguments."""
        default_mesh_arguments = {"mesh_size_frac": 0.05, "mesh_size_bound": 0.05}
        return self.params.get("mesh_arguments", default_mesh_arguments)

    def set_md_grid(self) -> None:
        """Set mixed-dimensional grid."""
        self.mdg = self.fracture_network.mesh(self.mesh_arguments())
        domain = self.fracture_network.domain
        if domain is not None and domain.is_boxed:
            self.domain = domain


# -----> Balance equations
class ManuPoroMechMassBalance(mass.MassBalanceEquations):
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

        # Add up contribution of internal and external sources of fluid
        fluid_sources = internal_sources + external_sources
        fluid_sources.set_name("Fluid source")

        """
        The following is a "hack" to include cross term missing from the dt operator.
        This reduces the discretization error significantly, but we don't include it
        since the purpose of this verification setup is to be used in functional
        tests to check the correct *default* behaviour of the models.

        Add cross term missing from dt operator relative to proper application of
        the product rule.

        .. code:: python

            rho = self.fluid_density(subdomains)
            phi = self.volume_integral(self.porosity(subdomains), subdomains, dim=1)
            dt_op = pp.ad.time_derivatives.dt
            dt = pp.ad.Scalar(self.time_manager.dt, name="delta_t")
            prod = dt_op(rho, dt) * dt_op(phi, dt)

        """

        return fluid_sources  # - prod


class ManuPoroMechMomentumBalance(momentum.MomentumBalanceEquations):
    """Modify momentum balance to account for time-dependent body force."""

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force."""

        external_sources = pp.ad.TimeDependentArray(
            name="source_mechanics",
            subdomains=self.mdg.subdomains(),
            previous_timestep=True,
        )

        return external_sources


class ManuPoroMechEquations(
    ManuPoroMechMassBalance,
    ManuPoroMechMomentumBalance,
):
    """Mixer class for modified poromoechanics equations."""

    def set_equations(self):
        """Set the equations for the modified poromechanics problem.

        Call both parent classes' `set_equations` methods.

        """
        ManuPoroMechMassBalance.set_equations(self)
        ManuPoroMechMomentumBalance.set_equations(self)


# -----> Solution strategy
class ManuPoroMechSolutionStrategy(poromechanics.SolutionStrategyPoromechanics):
    """Solution strategy for the verification setup."""

    exact_sol: ManuPoroMechExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants
    """Object containing the fluid constants."""

    plot_results: Callable
    """Method for plotting results. Usually provided by the mixin class
    :class:`SetupUtilities`.

    """

    results: list[ManuPoroMechSaveData]
    """List of SaveData objects."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

        self.exact_sol: ManuPoroMechExactSolution
        """Exact solution object."""

        self.results: list[ManuPoroMechSaveData] = []
        """Results object that stores exact and approximated solutions and errors."""

        self.flux_variable: str = "darcy_flux"
        """Keyword to access the Darcy fluxes."""

        self.stress_variable: str = "poroelastic_force"
        """Keyword to access the poroelastic force."""

    def set_materials(self):
        """Set material parameters."""
        super().set_materials()

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ManuPoroMechExactSolution(self)

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        mech_source = self.exact_sol.mechanics_source(sd=sd, time=t)
        data[pp.STATE]["source_mechanics"] = mech_source

        # Flow source
        flow_source = self.exact_sol.flow_source(sd=sd, time=t)
        data[pp.STATE]["source_flow"] = flow_source

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()


# -----> Mixer class
class ManuPoroMechSetup(  # type: ignore[misc]
    UnitSquareTriangleGrid,
    ManuPoroMechEquations,
    ManuPoroMechSolutionStrategy,
    ManuPoroMechUtils,
    ManuPoroMechDataSaving,
    poromechanics.Poromechanics,
):
    """
    Mixer class for the two-dimensional non-linear poromechanics verification setup.

    """
