"""
This module contains the implementation of Mandel's consolidation problem. The problem
is discretized using MPFA/MPSA-FV in space and backward Euler in time. For further
details on Mandel's problem, see [1 – 5]. For the implementation details, see [6].

References:

    - [1] Mandel, J.: Consolidation des sols (étude mathématique). Geotechnique.
      3(7), 287–299 (1953).

    - [2] Cheng, A.H.-D., Detournay, E.: A direct boundary element method for plane
     strain poroelasticity. Int. J. Numer. Anal. Methods Geomech. 12(5), 551–572 (1988).

    - [3] Abousleiman, Y., Cheng, A. D., Cui, L., Detournay, E., & Roegiers, J. C.
      (1996). Mandel's problem revisited. Geotechnique, 46(2), 187-195.

    - [4] Mikelić, A., Wang, B., & Wheeler, M. F. (2014). Numerical convergence study of
      iterative coupling for coupled flow and geomechanics. Computational Geosciences,
      18(3), 325-341.

    - [5] Borregales, M., Kumar, K., Radu, F. A., Rodrigo, C., & Gaspar, F. J. (2019). A
      partially parallel-in-time fixed-stress splitting method for Biot’s consolidation
      model. Computers & Mathematics with Applications, 77(6), 1466-1478.

    - [6] Keilegavlen, E., Berge, R., Fumagalli, A. et al. PorePy: an open-source
      software for simulation of multiphysics processes in fractured porous media.
      Comput Geosci 25, 243–265 (2021).

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sps

import porepy as pp
import porepy.models.fluid_mass_balance as mass
import porepy.models.poromechanics as poromechanics
from porepy.applications.classic_models.biot import BiotPoromechanics
from porepy.applications.verification_setups.verification_utils import (
    VerificationUtils,
    VerificationDataSaving,
)
from porepy.applications.verification_setups.manu_poromech_nofrac import (
    ManuPoroMechSaveData,
    ModifiedDataSavingMixin,
)

# PorePy typings
number = pp.number
grid = pp.GridLike

# Physical parameters for the verification setup
mandel_solid_constants: dict[str, number] = {
    "lame_lambda": 1.65e9,  # [Pa]
    "shear_modulus": 2.475e9,  # [Pa]
    "specific_storage": 6.0606e-11,  # [Pa^-1]
    "permeability": 9.869e-14,  # [m^2]
    "biot_coefficient": 1.0,  # [-]
}

mandel_fluid_constants: dict[str, number] = {
    "density": 1e3,  # [kg * m^3]
    "viscosity": 1e-3,  # [Pa * s]
}


# -----> Data-saving
@dataclass
class MandelSaveData(ManuPoroMechSaveData):
    """Class to store relevant data from the verification setup."""

    approx_consolidation_degree: Optional[tuple[number]] = None
    """Approximated degree of consolidation. First element of the tuple corresponds 
    to the horizontal and second to the vertical degrees of consolidation.

    """

    exact_consolidation_degree: Optional[number] = None
    """Exact degree of consolidation."""

    error_consolidation_degree: Optional[tuple[number]] = None
    """Absolute error for the degree of consolidation. First element is the error in 
    the horizontal direction. Second element is the error in the vertical direction.
    
    """


class MandelDataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    def collect_data(self) -> MandelSaveData:
        """Collect the data from the verification setup.

        Returns:
            MandelSaveData object containing the results of the verification for the
            current time.

        """

        # Retrieve information from setup
        mdg: pp.MixedDimensionalGrid = self.mdg
        sd: pp.Grid = mdg.subdomains()[0]
        data: dict = mdg.subdomain_data(sd)
        t: number = self.time_manager.time
        p_name: str = self.pressure_variable
        u_name: str = self.displacement_variable

        # Instantiate data class
        out = MandelSaveData()

        # Time



class MandelUtilities:
    """Mixin class that provides useful utility methods for the verification setup."""

    domain_boundary_sides: Callable[[pp.Grid], pp.bounding_box.DomainSides]
    """Named tuple containing the boundary sides indices."""

    exact_sol: MandelExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants

    fluid_diffusivity: Callable
    """Fluid diffusivity in `m^2 * s^{-1}`."""

    mdg: pp.MixedDimensionalGrid

    params: dict
    """Model parameters dictionary."""

    results: list[SaveData]

    solid: pp.SolidConstants

    time_manager: pp.TimeManager

    # -----> Non-dimensionalization methods
    def nondim_t(self, t: Union[float, int, np.ndarray]) -> float:
        """Non-dimensionalized time.

        Parameters:
            t: Time in seconds.

        Returns:
            Dimensionless time for the given time ``t``.

        """
        a, _ = self.params.get("domain_size", (100, 10)) # [m]
        c_f = self.fluid_diffusivity()  # [m^2 * s^{-1}]
        return (t * c_f) / (a**2)

    def nondim_x(self, x: np.ndarray) -> np.ndarray:
        """Nondimensionalized length in the horizontal direction.

        Parameters:
            x: horizontal length in meters.

        Returns:
            Dimensionless horizontal length with ``shape=(num_points, )``.

        """
        a, _ = self.params.get("domain_size", (100, 10))  # [m]
        return x / a

    def nondim_y(self, y: np.ndarray) -> np.ndarray:
        """Non-dimensionalized length in the vertical direction.

        Parameters:
            y: vertical length in meters.

        Returns:
            Dimensionless vertical length with ``shape=(num_points, )``.

        """
        _, b = self.params.get("domain_size", (100, 10))  # [m]
        return y / b

    def nondim_p(self, p: np.ndarray) -> np.ndarray:
        """Non-dimensionalized pressure.

        Parameters:
            p: Fluid pressure in Pascals.

        Returns:
            Dimensionaless pressure with ``shape=(num_points, )``.

        """
        a, _ = self.params.get("domain_size", (100, 10))  # [m]
        F = self.params.get("vertical_load", 6e8)  # [N * m^{-1}]
        return p / (F * a)

    def nondim_flux(self, qx: np.ndarray) -> np.ndarray:
        """Non-dimensionalized horizontal component of the specific discharge vector.

        Parameters:
            qx: Horizontal component of the specific discharge in `m * s^{-1}`.

        Returns:
            Dimensionless horizontal component of the specific discharge with
            ``shape=(num_points, )``.

        """
        k = self.solid.permeability()  # [m^2]
        F = self.params.get("vertical_load", 6e8)  # [N * m^{-1}]
        mu = self.fluid.viscosity()  # [Pa * s]
        a, _ = self.params.get("domain_size", (100, 10))  # [m]
        factor = (F * k) / (mu * a**2)  # [m * s^{-1}]
        return qx / factor

    def nondim_stress(self, syy: np.ndarray) -> np.ndarray:
        """Non-dimensionalized vertical component of the stress tensor.

        Parameters:
            syy: Vertical component of the stress tensor.

        Returns:
            Dimensionless vertical component of the stress tensor with
            ``shape=(num_points, )``.

        """
        a, _ = self.params.get("domain_size", (100, 10))  # [m]
        F = self.params.get("vertical_load", 6e8)  # [N * m^{-1}]
        return syy / (F / a)

    # -----> Post-processing methods
    def south_cells(self) -> np.ndarray:
        """Get indices of cells that are adjacent to the South boundary."""
        sd = self.mdg.subdomains()[0]
        sides = self.domain_boundary_sides(sd)
        south_idx = np.where(sides.south)[0]
        return sd.signs_and_cells_of_boundary_faces(south_idx)[1]

    def east_cells(self) -> np.ndarray:
        """Get indices of cells that are adjacent to the East boundary."""
        sd = self.mdg.subdomains()[0]
        sides = self.domain_boundary_sides(sd)
        east_idx = np.where(sides.east)[0]
        return sd.signs_and_cells_of_boundary_faces(east_idx)[1]

    # -----> Plotting methods
    def plot_results(self):
        """Ploting results."""
        num_lines = len(self.time_manager.schedule) - 1
        cmap = mcolors.ListedColormap(plt.cm.tab20.colors[: num_lines])

        self._plot_pressure(color_map=cmap)
        self._plot_horizontal_displacement(color_map=cmap)
        self._plot_vertical_displacement(color_map=cmap)
        self._plot_horizontal_flux(color_map=cmap)
        self._plot_vertical_stress(color_map=cmap)
        self._plot_consolidation_degree()

    def _plot_pressure(self, color_map: mcolors.ListedColormap) -> None:
        """Plot non-dimensional pressure profiles.

        Parameters:
            color_map: listed color map object.

        """
        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]
        south_cells = self.south_cells()
        a, _ = self.params.get("domain_size", (100, 10))
        x_ex = np.linspace(0, a, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, result in enumerate(self.results):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_p(self.exact_sol.pressure_profile(x_ex, result.time)),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xc[south_cells]),
                self.nondim_p(result.approx_pressure[south_cells]),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(result.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional horizontal distance, $x ~ a^{-1}$", fontsize=13)
        ax.set_ylabel(r"Non-dimensional pressure, $p ~ a ~ F^{-1}$", fontsize=13)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        plt.subplots_adjust(right=0.7)
        plt.show()

    def _plot_horizontal_displacement(self, color_map: mcolors.ListedColormap) -> None:
        """Plot non-dimensional horizontal displacement profiles.

        Parameters:
            color_map: listed color map object.

        """
        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]
        south_cells = self.south_cells()
        a, _ = self.params.get("domain_size", (100, 10))
        x_ex = np.linspace(0, a, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, result in enumerate(self.results):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_x(
                    self.exact_sol.horizontal_displacement_profile(x_ex, result.time)
                ),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xc[south_cells]),
                self.nondim_x(result.approx_displacement[::2][south_cells]),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(result.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional horizontal distance, $x ~ a^{-1}$", fontsize=13)
        ax.set_ylabel(
            r"Non-dimensional horizontal displacement, $u_x ~ a^{-1}$", fontsize=13
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        plt.show()

    def _plot_vertical_displacement(self, color_map: mcolors.ListedColormap) -> None:
        """Plot non-dimensional vertical displacement profiles.

        Parameters:
            color_map: listed color map object.

        """
        sd = self.mdg.subdomains()[0]
        yc = sd.cell_centers[1]
        east_cells = self.east_cells()
        _, b = self.params.get("domain_size", (100, 10))
        y_ex = np.linspace(0, b, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, result in enumerate(self.results):
            ax.plot(
                self.nondim_y(y_ex),
                self.nondim_y(
                    self.exact_sol.vertical_displacement_profile(y_ex, result.time)
                ),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_y(yc[east_cells]),
                self.nondim_y(result.approx_displacement[1::2][east_cells]),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(result.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional vertical distance, $y ~ b^{-1}$", fontsize=13)
        ax.set_ylabel(
            r"Non-dimensional vertical displacement, $u_y ~ b^{-1}$", fontsize=13
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        plt.show()

    def _plot_horizontal_flux(self, color_map: mcolors.ListedColormap) -> None:
        """Plot non-dimensional horizontal Darcy flux profiles.

        Parameters:
            color_map: listed color map object.

        """
        sd = self.mdg.subdomains()[0]
        xf = sd.face_centers[0]
        nx = sd.face_normals[0]
        sides = self.domain_boundary_sides(sd)
        south_cells = self.south_cells()
        faces_of_south_cells = sps.find(sd.cell_faces.T[south_cells])[1]
        south_faces = np.where(sides.south)[0]
        int_faces_of_south_cells = np.setdiff1d(faces_of_south_cells, south_faces)

        a, _ = self.params.get("domain_size", (100, 10))
        x_ex = np.linspace(0, a, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, result in enumerate(self.results):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_flux(
                    self.exact_sol.horizontal_velocity_profile(x_ex, result.time)
                ),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xf[int_faces_of_south_cells]),
                self.nondim_flux(
                    result.approx_flux[int_faces_of_south_cells]
                    / nx[int_faces_of_south_cells]
                ),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(result.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional horizontal distance, $x ~ a^{-1}$", fontsize=13)
        ax.set_ylabel(
            r"Non-dimensional horizontal flux, "
            r"$q_x ~ \mu_f ~ a^2 ~ F^{-1} ~ k^{-1}$",
            fontsize=13,
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        plt.show()

    def _plot_vertical_stress(self, color_map: mcolors.ListedColormap) -> None:
        """Plot non-dimensional vertical stress profiles.

        Parameters:
            color_map: listed color map object.

        """
        sd = self.mdg.subdomains()[0]
        xf = sd.face_centers[0]
        sides = self.domain_boundary_sides(sd)
        south_faces = sides.south
        ny = sd.face_normals[1]

        a, b = self.params.get("domain_size", (100, 10))
        x_ex = np.linspace(0, a, 400)

        # Vertical stress plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, result in enumerate(self.results):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_stress(
                    self.exact_sol.vertical_stress_profile(x_ex, result.time)
                ),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xf[south_faces]),
                self.nondim_stress(
                    result.approx_force[1 :: sd.dim][south_faces]
                    / ny[south_faces]
                ),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(result.time), 5)}",
            )
        ax.set_xlabel(
            r"Non-dimensional horizontal distance, $ x ~ a^{-1}$", fontsize=13
        )
        ax.set_ylabel(
            r"Non-dimensional vertical poroelastic stress,"
            r" $\sigma_{yy} ~ a ~ F^{-1}$",
            fontsize=13,
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        plt.show()

    def _plot_consolidation_degree(self):
        """Plot degree of consolidation as a function of time."""

        # Retrieve data
        a, _ = self.params.get("domain_size", (100, 10))  # [m]
        c_f = self.fluid_diffusivity()  # [m^2 * s^{-1}]

        # Generate exact consolidation times
        tau_0 = 1e-3
        tau_1 = 1e-2
        tau_2 = 1e0
        tau_3 = 1e1

        t = lambda tau: tau * a**2 / c_f

        interval_0 = np.linspace(t(tau_0), t(tau_1), 100)
        interval_1 = np.linspace(t(tau_1), t(tau_2), 100)
        interval_2 = np.linspace(t(tau_2), t(tau_3), 100)
        ex_times = np.concatenate((interval_0, interval_1))
        ex_times = np.concatenate((ex_times, interval_2))

        ex_consol_degree = np.array(
            [self.exact_sol.degree_of_consolidation(t) for t in ex_times]
        )

        # Numerical consolidation degrees
        num_times = self.time_manager.schedule[1:]
        num_consol_degree_x = np.array(
            [sol.num_consol_degree[0] for sol in self.results]
        )
        num_consol_degree_y = np.array(
            [sol.num_consol_degree[1] for sol in self.results]
        )

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.semilogx(
            self.nondim_t(ex_times),
            ex_consol_degree,
            alpha=0.5,
            color="black",
            linewidth=2,
            label="Exact",
        )
        ax.semilogx(
            self.nondim_t(num_times),
            num_consol_degree_x,
            marker="s",
            linewidth=0,
            markerfacecolor="none",
            markeredgewidth=2,
            color="blue",
            markersize=10,
            label="Horizontal MPFA/MPSA-FV",
        )
        ax.semilogx(
            self.nondim_t(num_times),
            num_consol_degree_y,
            marker="+",
            markeredgewidth=2,
            linewidth=0,
            color="red",
            markersize=10,
            label="Vertical MPFA/MPSA-FV",
        )
        ax.set_xlabel(r"Non-dimensional time, $t ~ c_f ~ a^{-2}$", fontsize=13)
        ax.set_ylabel(
            r"Degree of consolidation," r" $U(t)$",
            fontsize=13,
        )
        ax.grid()
        ax.legend(fontsize=12)
        plt.show()


class MandelGeometry(pp.ModelGeometry):
    """Class for setting up the rectangular geometry."""

    params: dict
    """Simulation model parameters."""

    fracture_network: pp.FractureNetwork2d
    """Fracture network. Empty in this case."""

    def set_fracture_network(self) -> None:
        """Set fracture network. Unit square with no fractures."""
        a, b = self.params.get("domain_size", (100, 10))
        domain = {"xmin": 0.0, "xmax": a, "ymin": 0.0, "ymax": b}
        self.fracture_network = pp.FractureNetwork2d(None, None, domain)

    def mesh_arguments(self) -> dict:
        """Set mesh arguments."""
        default_mesh_arguments = {"mesh_size_frac": 2, "mesh_size_bound": 2}
        return self.params.get("mesh_arguments", default_mesh_arguments)

    def set_md_grid(self) -> None:
        """Set mixed-dimensional grid."""
        self.mdg = self.fracture_network.mesh(self.mesh_arguments())
        domain = self.fracture_network.domain
        if isinstance(domain, np.ndarray):
            assert domain.shape == (2, 2)
            self.domain_bounds: dict[str, float] = {
                "xmin": domain[0, 0],
                "xmax": domain[1, 0],
                "ymin": domain[0, 1],
                "ymax": domain[1, 1],
            }
        else:
            assert isinstance(domain, dict)
            self.domain_bounds = domain


class MandelExactSolution:
    """Exact solutions to Mandel's problem."""

    def __init__(self, setup):
        """Constructor of the class."""
        self.setup = setup

        # For convinience, store the approximated roots
        self.roots: np.ndarray = self.approximate_roots()

    def approximate_roots(self) -> np.ndarray:
        """Approximate roots of infinte series using the bisection method.

        Returns:
            Array of ``shape=(num_roots, )`` containing the approximated roots.

        Note:
            We solve

            .. math::

               f(x) = \\tan{x} - \\frac{1 - \\nu}{\\nu_u - \\nu} x = 0

            numerically to get all positive solutions to the equation. These are used
            to compute the infinite series associated with the exact solutions.
            Experience has shown that 200 roots are enough to achieve accurate results.
            We find the roots using the bisection method.

            Thanks to Manuel Borregales who helped us with the implementation of this
            part of the code.

        """
        # Retrieve physical data
        nu_s = self.setup.poisson_coefficient()
        nu_u = self.setup.undrained_poisson_coefficient()

        # Define algebraic function
        def f(x):
            y = np.tan(x) - ((1 - nu_s) / (nu_u - nu_s)) * x
            return y

        n_series = self.setup.params.get("number_of_roots", 200)
        a_n = np.zeros(n_series)  # initialize roots
        x0 = 0  # initial point
        for i in range(n_series):
            a_n[i] = opt.bisect(
                f,  # function
                x0 + np.pi / 4,  # left point
                x0 + np.pi / 2 - 2.2204e-9,  # right point
                xtol=1e-30,  # absolute tolerance
                rtol=1e-14,  # relative tolerance
            )
            x0 += np.pi  # apply a phase change of pi to get the next root

        return a_n

    def pressure_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact fluid pressure in `Pa`.

        Parameters:
            x: Points in the horizontal axis in meters.
            t: Time in seconds.

        Returns:
            Exact pressure solution with ``shape=(num_points, )``.

        """
        # Retrieve physical data
        F = self.setup.params.get("vertical_load", 6e8)
        B = self.setup.skempton_coefficient()
        nu_u = self.setup.undrained_poisson_coefficient()
        c_f = self.setup.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.setup.params("domain_size", (100, 10))

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Exact pressure
        if t == 0:  # initial condition has its own expression

            p = ((F * B * (1 + nu_u)) / (3 * a)) * np.ones_like(x)

        else:

            c0 = (2 * F * B * (1 + nu_u)) / (3 * a)

            p_sum_0 = np.sum(
                ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * (np.cos((aa_n * x) / a) - np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )

            p = c0 * p_sum_0

        return p

    def horizontal_displacement_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact horizontal displacement in `m`.

        Parameters:
            x: Points in the horizontal axis in meters.
            t: Time in seconds.

        Returns:
            Exact horizontal displacement with ``shape=(num_points, )``.

        """
        # Retrieve physical data
        F = self.setup.params.get("vertical_load", 6e8)
        nu_s = self.setup.poisson_coefficient()
        nu_u = self.setup.undrained_poisson_coefficient()
        mu_s = self.setup.solid.shear_modulus()
        c_f = self.setup.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.setup.params("domain_size", (100, 10))

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Exact horizontal displacement
        if t == 0:  # initial condition has its own expression

            ux = ((F * nu_u) / (2 * mu_s * a)) * x

        else:

            cx0 = (F * nu_s) / (2 * mu_s * a)
            cx1 = -((F * nu_u) / (mu_s * a))
            cx2 = F / mu_s

            ux_sum1 = np.sum(
                (np.sin(aa_n) * np.cos(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )

            ux_sum2 = np.sum(
                (np.cos(aa_n) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * np.sin((aa_n * x) / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )

            ux = (cx0 + cx1 * ux_sum1) * x + cx2 * ux_sum2

        return ux

    def vertical_displacement_profile(self, y: np.ndarray, t: number) -> np.ndarray:
        """Exact vertical displacement in `m`.

        Parameters:
            y: Points in the horizontal axis in meters.
            t: Time in seconds.

        Returns:
            Exact vertical displacement with ``shape=(num_points, )``.

        """
        # Retrieve physical data
        F = self.setup.params.get("vertical_load", 6e8)
        nu_s = self.setup.poisson_coefficient()
        nu_u = self.setup.undrained_poisson_coefficient()
        mu_s = self.setup.solid.shear_modulus()
        c_f = self.setup.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.setup.params("domain_size", (100, 10))

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Exact vertical displacement
        if t == 0:  # initial condition has its own expression

            uy = ((-F * (1 - nu_u)) / (2 * mu_s * a)) * y

        else:

            cy0 = (-F * (1 - nu_s)) / (2 * mu_s * a)
            cy1 = F * (1 - nu_u) / (mu_s * a)

            uy_sum1 = np.sum(
                ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )

            uy = (cy0 + cy1 * uy_sum1) * y

        return uy

    def horizontal_velocity_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact horiztonal specific discharge in `m * s^{-1}`.

        Note that for Mandel's problem, the vertical specific discharge is zero.

        Parameters:
            x: Points in the horizontal axis in `m`.
            t: Time in `s`.

        Returns:
            Exact specific discharge for the given time ``t``. The returned array has
             ``shape=(num_points, )``.

        """
        # Retrieve physical data
        F = self.setup.params.get("vertical_load", 6e8)
        B = self.setup.skempton_coefficient()
        k = self.setup.solid("permeability")
        mu_f = self.setup.fluid("viscosity")
        nu_u = self.setup.undrained_poisson_coefficient()
        c_f = self.setup.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.setup.params("domain_size", (100, 10))

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Compute horizontal specific discharge
        if t == 0:

            qx = [np.zeros_like(x), np.zeros_like(x)]  # zero initial discharge

        else:

            c0 = (2 * F * B * k * (1 + nu_u)) / (3 * mu_f * a**2)

            qx_sum0 = np.sum(
                (aa_n * np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.sin(aa_n * x / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )

            qx = c0 * qx_sum0

        return qx

    def vertical_stress_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact vertical stress in `Pa`.

        Note that for Mandel's problem, all the other components of the stress tensor
        are zero.

        Parameters:
            x: Points in the horizontal axis in meters.
            t: Time in seconds.

        Returns:
            Exact vertical component of the symmetric stress tensor with
            ``shape=(num_points, )``.

        """
        # Retrieve physical data
        F = self.setup.params.get("vertical_load", 6e8)
        nu_s = self.setup.poisson_coefficient()
        nu_u = self.setup.undrained_poisson_coefficient()
        c_f = self.setup.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.setup.params("domain_size", (100, 10))

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Compute exact stress tensor

        if t == 0:  # vertical stress at t = 0 has a different expression

            syy = -F / a * np.ones_like(x)

        else:

            c0 = -F / a
            c1 = (-2 * F * (nu_u - nu_s)) / (a * (1 - nu_s))
            c2 = 2 * F / a

            syy_sum1 = np.sum(
                (np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.cos(aa_n * x / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )

            syy_sum2 = np.sum(
                (np.sin(aa_n) * np.cos(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )

            syy = c0 + c1 * syy_sum1 + c2 * syy_sum2

        return syy

    def degree_of_consolidation(self, t: number) -> number:
        """Exact degree of consolidation [-].

        Note that for Mandel's problem, the degrees of consolidation in the
        horizontal and vertical axes are identical.

        Parameters:
              t: Time in seconds.

        Returns:
              Exact degree of consolidation for a given time ``t``.

        """
        # Retrieve physical and geometric data
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]
        nu_s = self.setup.poisson_coefficient()  # [-]
        mu_s = self.setup.solid.shear_modulus()  # [Pa]
        F = self.setup.params.get("vertical_load", 6e8)  # [N * m^{-1}]
        a, b = self.setup.params("domain_size", (100, 10))  # ([m], [m])

        # Vertical displacement on the north boundary at time `t`
        uy_b_t = self.vertical_displacement_profile(np.array([b]), t)

        # Initial vertical displacement on the north boundary
        uy_b_0 = (-F * b * (1 - nu_u)) / (2 * mu_s * a)

        # Vertical displacement on the north boundary at time `infinity`
        uy_b_inf = (-F * b * (1 - nu_s)) / (2 * mu_s * a)

        # Degree of consolidation
        consolidation_degree = (uy_b_t - uy_b_0) / (uy_b_inf - uy_b_0)

        return consolidation_degree

    def pressure(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact pressure [Pa] at the cell centers.

        Parameters:
            sd: Grid.
            t: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact pressures at
            the cell centers at the given time ``t``.

        """
        return self.pressure_profile(sd.cell_centers[0], t)

    def displacement(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact displacement [m] at the cell centers.

        Parameters:
            sd: Grid.
            t: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells * 2, )`` containing the exact displacement at
            the cell centers at the given time ``t``. The returned array is given in
            flattened vector format.

        """
        ux = self.horizontal_displacement_profile(sd.cell_centers[0], t)
        uy = self.vertical_displacement_profile(sd.cell_centers[1], t)
        u = np.stack((ux, uy)).ravel("F")
        return u

    def flux(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact Darcy flux [m^3 * s^{-1}] at the face centers.

        Parameters:
            sd: Grid.
            t: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact Darcy flux at
            the face centers at the given time ``t``.

        """
        qx = self.horizontal_velocity_profile(sd.face_centers[0], t)
        nx = sd.face_normals[0]
        darcy_flux = qx * nx
        return darcy_flux

    def poroelastic_force(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact poroelastic force [N] at the face centers.

        Parameters:
            sd: Grid.
            t: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces * 2, )`` containing the exact poroelastic
            force at the face centers at the given time ``t``.

        """
        syy = self.vertical_stress_profile(sd.face_centers[0], t)
        ny = sd.face_normals[1]
        force_y = syy * ny
        force_x = np.zeros_like(force_y)
        force = np.stack((force_x, force_y)).ravel("F")
        return force


@dataclass
class MandelSolution:
    """Data class to store variables of interest for Mandel's problem"""

    def __init__(self, setup: "Mandel"):
        """Data class constructor.
        Args:
            setup : Mandel's model application.
        """
        sd = setup.mdg.subdomains()[0]
        data = setup.mdg.subdomain_data(sd)
        p_var = setup.scalar_variable
        u_var = setup.displacement_variable
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]
        xf = sd.face_centers[0]
        t = setup.time_manager.time

        # Time variables
        self.time = t

        # Pressure variables
        self.num_pressure = data[pp.STATE][p_var]
        self.ex_pressure = setup.exact_pressure(xc, t)

        # Flux variables
        self.num_flux = setup.numerical_flux()
        self.ex_flux = setup.velocity_to_flux(setup.exact_velocity(xf, t))

        # Displacement variables
        self.num_displacement = data[pp.STATE][u_var]
        self.num_displacement_x = self.num_displacement[:: sd.dim]
        self.num_displacement_y = self.num_displacement[1 :: sd.dim]

        self.ex_displacement_x = setup.exact_horizontal_displacement(xc, t)
        self.ex_displacement_y = setup.exact_vertical_displacement(yc, t)
        self.ex_displacement = setup.ravel(
            [self.ex_displacement_x, self.ex_displacement_y]
        )

        # Traction variables
        self.num_traction = setup.numerical_traction()
        self.ex_traction = setup.ravel(
            setup.stress_to_traction(setup.exact_stress(xf, t))
        )

        # Consolidation degree
        self.num_consol_degree = setup.numerical_consolidation_degree()
        self.ex_consol_degree = setup.exact_degree_of_consolidation(t)

        # Error variables
        self.pressure_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_pressure,
            approx_array=self.num_pressure,
            is_scalar=True,
            is_cc=True,
        )
        self.flux_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_flux,
            approx_array=self.num_flux,
            is_scalar=True,
            is_cc=False,
        )
        self.displacement_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_displacement,
            approx_array=self.num_displacement,
            is_scalar=False,
            is_cc=True,
        )
        self.traction_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_traction,
            approx_array=self.num_traction,
            is_scalar=False,
            is_cc=False,
        )


class MandelSolutionStrategy(poromechanics.SolutionStrategyPoromechanics):
    """Solution strategy for Mandel's problem."""

    exact_sol: MandelExactSolution
    """Exact solution object."""

    def __init__(self, params: dict):
        """Constructor of the class."""
        super().__init__(params)

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().__init__()

        # Sanity check
        assert self.solid.biot_coefficient() == 1

        # Instantiate exact solution object after the material constants have been set
        self.exact_sol = MandelExactSolution(self)

    def initial_condition(self) -> None:
        """Modify default initial conditions.

        Note:
            Initial conditions are given in 10.1007/s10596-013-9393-8.

        """
        super().initial_condition()

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        p_name = self.pressure_variable
        u_name = self.displacement_variable

        # Set initial pressure
        data[pp.STATE][p_name] = self.exact_sol.pressure(sd, 0)
        data[pp.STATE][pp.ITERATE][p_name] = data[pp.STATE][p_name].copy()

        # Set initial displacement
        data[pp.STATE][u_name] = self.exact_sol.displacement(sd, 0)
        data[pp.STATE][pp.ITERATE][u_name] = data[pp.STATE][u_name].copy()

    def before_nonlinear_loop(self) -> None:
        """Update values of mechanical boundary conditions."""
        super().before_nonlinear_loop()


class ModifiedBoundaryConditionsMechanicsTimeDependent(
    poromechanics.BoundaryConditionsMechanicsTimeDependent
):

    bc_values_mechanics_key: str
    """Keyword for accessing the boundary values for the mechanical subproblem."""

    domain_boundary_sides: Callable[[pp.Grid], pp.bounding_box.DomainSides]
    """Named tuple containing the boundary sides indices."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    params: dict
    """Parameter dictionary of the verification setup."""

    stress_keyword: str
    """Keyword for the mechanical subproblem."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions.

        For Mandel's problem, we set all boundary sides as rollers, except the East
        side which is kept at stress-free conditions.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition object on the grid ``sd``.

        """


class Mandel(pp.ContactMechanicsBiot):
    """Parent class for Mandel's problem.
    Examples:
        .. code:: python
            # Import modules
            import porepy as pp
            from time import time
            # Run setup
            tic = time()
            setup = Mandel({"plot_results": True})
            print("Simulation started...")
            pp.run_time_dependent_model(setup, setup.params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds.")
    """

    def __init__(self, params: dict):
        """Constructor of the Mandel class.
        Parameters:
            params: Dictionary containing mandatory and optional model parameters.
                Default physical parameters are takem from
                https://link.springer.com/article/10.1007/s10596-013-9393-8.
                Optional parameters are:
                - 'alpha_biot' : Biot constant (int or float). Default is 1.0.
                - 'domain_size' : Size of the domain in `m` (tuple of int or float).
                  First element is the length and second the height.
                  Default is (100.0, 10.0).
                - 'lambda_lame' : Lamé parameter in `Pa` (int or float).
                  Default is 1.65e9.
                - 'mesh_size' : Mesh size in `m` (int or float). Default is 2.0.
                - 'mu_lame' : Lamé parameter in `m` (int or float). Default is 2.475E9.
                - 'number_of_roots' : Number of roots to approximate the exact
                  solutions (int). Default is 200.
                - 'permeability' : Permeability in `m^2` (int or float).
                  Default is 9.869e-14.
                - 'plot_results' : Whether to plot the results (bool). The resulting
                  plot is saved inside the `out` folder. Default is False.
                - 'storativity' : Storativity in `Pa^-1` (int or float).
                  Default is 6.0606e-11.
                - 'time_manager' : Time manager object (pp.TimeManager). Default is
                  pp.TimeManager(
                        schedule=[0, 1e1, 5e1, 1e2, 1e3, 5e3, 8e3, 1e4, 2e4, 3e4, 5e4],
                        dt_init=10,
                        constant_dt=True,
                  ).
                - 'use_ad' : Whether to use ad (bool). Must be set to True. Otherwise,
                  an error will be raised. Default is True.
                - 'vertical_load' : Applied vertical load in `N * m^-1` (int or float).
                  Default is 6e8.
                - 'viscosity' : Fluid viscosity in `Pa * s` (int or float).
                  Default is 1e-3.
        """

        def set_default_params(keyword: str, value: object) -> None:
            """
            Set default parameters if a keyword is absent in the `params` dictionary.
            Parameters:
                keyword: Parameter keyword, e.g., "alpha_biot".
                value: Value of `keyword`, e.g., 1.0.
            """
            if keyword not in params.keys():
                params[keyword] = value

        # Default parameters
        default_tm = pp.TimeManager(
            schedule=[0, 1e1, 5e1, 1e2, 1e3, 5e3, 8e3, 1e4, 2e4, 3e4, 5e4],
            dt_init=10,
            constant_dt=True,
        )

        default_params: list[tuple] = [
            ("alpha_biot", 1.0),  # [-]
            ("domain_size", (100.0, 10.0)),  # [m]
            ("lambda_lame", 1.65e9),  # [Pa]
            ("mesh_size", 2.0),  # [m]
            ("mu_lame", 2.475e9),  # [Pa]
            ("number_of_roots", 200),
            ("permeability", 9.869e-14),  # [m^2]
            ("plot_results", False),
            ("storativity", 6.0606e-11),  # [Pa^-1]
            ("time_manager", default_tm),  # all time-related variables must be in [s]
            ("use_ad", True),  # only `use_ad = True` is supported
            ("vertical_load", 6e8),  # [N * m^-1]
            ("viscosity", 1e-3),  # [Pa * s]
        ]

        # Set default values
        for key, val in default_params:
            set_default_params(key, val)
        super().__init__(params)

        # ad sanity check
        if not self.params["use_ad"]:
            raise ValueError("Model only valid when ad is used.")

        # Create a solution list to store variables
        self.solutions: list[MandelSolution] = []

    def create_grid(self) -> None:
        """Create a two-dimensional unstructured triangular grid."""
        lx, ly = self.params["domain_size"]
        mesh_size = self.params["mesh_size"]
        self.box = {"xmin": 0.0, "xmax": lx, "ymin": 0.0, "ymax": ly}
        network_2d = pp.FractureNetwork2d(None, None, self.box)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        self.mdg = network_2d.mesh(mesh_args)

    def _initial_condition(self) -> None:
        """Set up initial conditions.

        Note:
            Initial conditions are given by Eqs. (41) - (43) from
            10.1007/s10596-013-9393-8.

        """
        super()._initial_condition()
        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]
        data = self.mdg.subdomain_data(sd)

        # Set initial pressure
        data[pp.STATE][self.scalar_variable] = self.exact_pressure(xc, 0)
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = self.exact_pressure(xc, 0)

        # Set initial displacement
        ux0 = self.exact_horizontal_displacement(xc, 0)
        uy0 = self.exact_vertical_displacement(yc, 0)
        u0 = self.ravel([ux0, uy0])
        data[pp.STATE][self.displacement_variable] = u0
        data[pp.STATE][pp.ITERATE][self.displacement_variable] = u0

        # Store initial solution
        self.solutions.append(MandelSolution(self))

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.
        Parameters:
            sd: Subdomain grid.
        Returns:
            Scalar boundary condition representation.
        """
        # Define boundary regions
        sides = self._domain_boundary_sides(sd)
        east_bc = np.isin(sides.all_bf, np.where(sides.east)).nonzero()

        # All sides Neumann, except the East side which is Dirichlet
        bc_type = np.asarray(sides.all_bf.size * ["neu"])
        bc_type[east_bc] = "dir"

        bc = pp.BoundaryCondition(sd, faces=sides.all_bf, cond=list(bc_type))

        return bc

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define boundary condition types for the mechanics subproblem.
        Parameters:
            sd: Subdomain grid.
        Returns:
            Vectorial boundary condition representation.
        """
        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        super()._bc_type_mechanics(sd=sd)

        # Get boundary sides, retrieve data dict, and bc object
        sides = self._domain_boundary_sides(sd)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Stress-free
        bc.is_neu[:, sides.east] = True
        bc.is_dir[:, sides.east] = False

        # West side: Roller
        bc.is_neu[1, sides.west] = True
        bc.is_dir[1, sides.west] = False

        # North side: Roller
        bc.is_neu[0, sides.north] = True
        bc.is_dir[0, sides.north] = False

        # South side: Roller
        bc.is_neu[0, sides.south] = True
        bc.is_dir[0, sides.south] = False

        return bc

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem.
        Parameters:
            sd: Subdomain grid.
        Returns:
            Boundary condition values of ``shape=(sd.dim * sd.num_faces, )``.
        """
        # Retrieve boundary sides
        _, _, _, north, *_ = self._domain_boundary_sides(sd)

        # All zeros except vertical component of the north side
        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["vertical_load"]
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]

        # Retrieve geometrical data
        a, b = self.params["domain_size"]

        u0y = (-F * b * (1 - nu_u)) / (2 * mu_s * a)
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = u0y
        bc_values = bc_values.ravel("F")

        return bc_values

    def before_newton_loop(self) -> None:
        """Method to be called before entering the Newton loop."""
        super().before_newton_loop()
        # Update value of boundary conditions
        self.update_north_bc_values(self.time_manager.time)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Store solutions
        schedule = self.time_manager.schedule
        if any([np.isclose(self.time_manager.time, t_sch) for t_sch in schedule]):
            self.solutions.append(MandelSolution(self))

    def update_north_bc_values(self, t: Union[float, int]) -> None:
        """Update boundary condition value at the north boundary of the domain.
        Parameters:
            t: Time in `s`.
        Notes:
            The key `bc_values` from data[pp.PARAMETERS][self.mechanics_parameter_key]
            will be updated accordingly.
        """
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        kw_m = self.mechanics_parameter_key

        # Retrieve exact vertical displacement at the north boundary
        sides = self._domain_boundary_sides(sd)
        yf_north = sd.face_centers[1][sides.north]
        uy_north = self.exact_vertical_displacement(yf_north, t)

        # Update values
        data[pp.PARAMETERS][kw_m]["bc_values"][1 :: sd.dim][sides.north] = uy_north

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params["plot_results"]:
            self.plot_results()

    # Physical parameters
    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Override permeability value [m^2].
        Parameters:
            sd: Subdomain grid.
        Returns:
            Permeability.
        """
        return self.params["permeability"] * np.ones(sd.num_cells)

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Override stiffness tensor.
        Parameters:
            sd: Subdomain grid.
        Returns:
            Fourth order tensorial representation of the stiffness tensor.
        """
        lam = (self.params["lambda_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        mu = (self.params["mu_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        return pp.FourthOrderTensor(mu, lam)

    def _viscosity(self, sd: pp.Grid) -> np.ndarray:
        """Override fluid viscosity values [Pa * s].
        Parameters:
            sd: Subdomain grid.
        Returns:
            Viscosity.
        """
        return self.params["viscosity"] * np.ones(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Override storativity value of the porous medium [1/Pa].
        Parameters:
            sd: Subdomain grid.
        Returns:
            Storativity.
        """
        return self.params["storativity"] * np.ones(sd.num_cells)

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Override value of Biot-Willis coefficient [-].
        Parameters:
            sd: Subdomain grid.
        Returns:
            Biot's coefficient.
        """
        return self.params["alpha_biot"] * np.ones(sd.num_cells)

    def bulk_modulus(self) -> float:
        """Set bulk modulus [Pa].
        Returns:
            Bulk modulus.
        """
        mu_s = self.params["mu_lame"]
        lambda_s = self.params["lambda_lame"]
        K_s = (2 / 3) * mu_s + lambda_s

        return K_s

    def young_modulus(self) -> float:
        """Set Young modulus [Pa].
        Returns:
            Young modulus.
        """
        mu_s = self.params["mu_lame"]
        K_s = self.bulk_modulus()
        E_s = mu_s * ((9 * K_s) / (3 * K_s + mu_s))

        return E_s

    def poisson_coefficient(self) -> float:
        """Set Poisson coefficient [-]
        Returns:
            Poisson coefficient.
        """
        mu_s = self.params["mu_lame"]
        K_s = self.bulk_modulus()
        nu_s = (3 * K_s - 2 * mu_s) / (2 * (3 * K_s + mu_s))

        return nu_s

    def undrained_bulk_modulus(self) -> float:
        """Set undrained bulk modulus [Pa].
        Returns:
            Undrained bulk modulus.
        """
        alpha_biot = self.params["alpha_biot"]
        K_s = self.bulk_modulus()
        S_m = self.params["storativity"]
        K_u = K_s + (alpha_biot**2) / S_m

        return K_u

    def skempton_coefficient(self) -> float:
        """Set Skempton's coefficient [-].
        Returns:
            Skempton's coefficent.
        """
        alpha_biot = self.params["alpha_biot"]
        K_u = self.undrained_bulk_modulus()
        S_m = self.params["storativity"]
        B = alpha_biot / (S_m * K_u)

        return B

    def undrained_poisson_coefficient(self) -> float:
        """Set Poisson coefficient under undrained conditions [-].
        Returns:
            Undrained Poisson coefficient.
        Notes:
            Expression taken from https://doi.org/10.1016/j.camwa.2018.09.005.
        """
        nu_s = self.poisson_coefficient()
        B = self.skempton_coefficient()
        nu_u = (3 * nu_s + B * (1 - 2 * nu_s)) / (3 - B * (1 - 2 * nu_s))

        return nu_u

    def fluid_diffusivity(self) -> float:
        """Set fluid diffusivity [m^2/s].
        Returns:
            Fluid diffusivity.
        """
        k_s = self.params["permeability"]
        B = self.skempton_coefficient()
        mu_s = self.params["mu_lame"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_f = self.params["viscosity"]
        c_f = (2 * k_s * (B**2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / (
            9 * mu_f * (1 - nu_u) * (nu_u - nu_s)
        )

        return c_f

    # -----> Methods related to exact solutions
    def approximate_roots(self) -> np.ndarray:
        """
        Approximate roots to f(x) = 0, where f(x) = tan(x) - ((1-nu)/(nu_u-nu)) x
        Note that we have to solve the above equation numerically to get all positive
        solutions to the equation. Later, we will use them to compute the infinite
        series associated with the exact solutions. Experience has shown that 200
        roots are enough to achieve accurate results.
        We find the roots using the bisection method. Thanks to Manuel Borregales
        who helped with the implementation of this part of the code. I have no
        idea what was the rationale behind the parameter tuning of the `bisect`
        method, but it seems to give good results.
        Returns:
            Approximated roots of `f(x) = 0` with ``shape=(num_roots, )``.
        """
        # Retrieve physical data
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()

        # Define algebraic function
        def f(x):
            y = np.tan(x) - ((1 - nu_s) / (nu_u - nu_s)) * x
            return y

        n_series = self.params.get("number_of_roots", 200)
        a_n = np.zeros(n_series)  # initializing roots array
        x0 = 0  # initial point
        for i in range(n_series):
            a_n[i] = opt.bisect(
                f,  # function
                x0 + np.pi / 4,  # left point
                x0 + np.pi / 2 - 10000000 * 2.2204e-16,  # right point
                xtol=1e-30,  # absolute tolerance
                rtol=1e-14,  # relative tolerance
            )
            x0 += np.pi  # apply a phase change of pi to get the next root

        return a_n

    def exact_pressure(self, x: np.ndarray, t: Union[float, int]) -> np.ndarray:
        """Exact pressure solution for a given time `t`.
        Parameters:
            x: Points in the horizontal axis.
            t: Time in seconds.
        Returns:
            Exact pressure solution with ``shape=(np, )``.
        """
        # Retrieve data
        F = self.params["vertical_load"]
        B = self.skempton_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Compute exact pressure
        if t == 0.0:  # initial condition has its own expression
            p = ((F * B * (1 + nu_u)) / (3 * a)) * np.ones_like(x)
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact p
            c0 = (2 * F * B * (1 + nu_u)) / (3 * a)
            p_sum_0 = np.sum(
                ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * (np.cos((aa_n * x) / a) - np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            p = c0 * p_sum_0

        return p

    def exact_horizontal_displacement(
        self, x: np.ndarray, t: Union[int, float]
    ) -> np.ndarray:
        """Exact horizontal displacement for a given time ``t``.
        Parameters:
            x: Points in the horizontal axis in `m`.
            t: Time in `s`
        Returns:
            Exact displacement in the horizontal direction with ``shape=(np, )``.
        """
        # Retrieve physical data
        F = self.params["vertical_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Determine displacements
        if t == 0:  # initial condition has its own expression
            ux = ((F * nu_u) / (2 * mu_s * a)) * x
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact horizontal displacement
            cx0 = (F * nu_s) / (2 * mu_s * a)
            cx1 = -((F * nu_u) / (mu_s * a))
            cx2 = F / mu_s
            ux_sum1 = np.sum(
                (np.sin(aa_n) * np.cos(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            ux_sum2 = np.sum(
                (np.cos(aa_n) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * np.sin((aa_n * x) / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            ux = (cx0 + cx1 * ux_sum1) * x + cx2 * ux_sum2
        return ux

    def exact_vertical_displacement(
        self, y: np.ndarray, t: Union[int, float]
    ) -> np.ndarray:
        """Exact vertical displacement for a given time ``t``.
        Parameters:
            y: Points in the horizontal axis in `m`.
            t: Time in `s`
        Returns:
            Exact displacement in the vertical direction with ``shape=(np, )``.
        """
        # Retrieve physical data
        F = self.params["vertical_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Determine displacements
        if t == 0.0:  # initial condition has its own expression
            uy = ((-F * (1 - nu_u)) / (2 * mu_s * a)) * y
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact vertical displacement
            cy0 = (-F * (1 - nu_s)) / (2 * mu_s * a)
            cy1 = F * (1 - nu_u) / (mu_s * a)
            uy_sum1 = np.sum(
                ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            uy = (cy0 + cy1 * uy_sum1) * y
        return uy

    def exact_velocity(
        self, x: np.ndarray, t: Union[float, int]
    ) -> list[np.ndarray, np.ndarray]:
        """Exact Darcy's velocity (specific discharge) in `m * s^{-1}`.
        For Mandel's problem, only the horizontal component in non-zero.
        Parameters:
            x: Points in the horizontal axis in `m`.
            t: Time in `s`.
        Returns:
            List of exact velocities for the given time ``t``. Each item of the list has
             ``shape=(np, )``.
        """
        # Retrieve physical data
        F = self.params["vertical_load"]
        B = self.skempton_coefficient()
        k = self.params["permeability"]
        mu_f = self.params["viscosity"]
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Compute specific discharge
        if (
            t == 0
        ):  # https://link.springer.com/content/pdf/10.1007/s10596-018-9736-6.pdf
            q = [np.zeros_like(x), np.zeros_like(x)]
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]

            c0 = (2 * F * B * k * (1 + nu_u)) / (3 * mu_f * a**2)
            qx_sum0 = np.sum(
                (aa_n * np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.sin(aa_n * x / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            qx = c0 * qx_sum0

            q = [qx, np.zeros_like(x)]

        return q

    def exact_stress(
        self,
        x: np.ndarray,
        t: Union[float, int],
    ) -> list[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]:
        """Exact stress tensor in `Pa`.
        In Mandel's problem, only the `yy` component of the stress tensor is non-zero.
        Parameters:
            x: Points in the horizontal axis in meters.
            t: Time in seconds.
        Returns:
            List of lists of arrays, representing the components of the exact symmetric
                stress tensor. Each item of the inner lists has ``shape=(np, )``.
        """
        # Retrieve physical data
        F = self.params["vertical_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Compute exact stress tensor

        sxx = np.zeros_like(x)  # sxx component of the stress is zero
        sxy = np.zeros_like(x)  # sxy components of the stress is zero
        syx = np.zeros_like(x)  # syx components of the stress is zero

        if t == 0:  # traction force at t = 0 has a different expression

            # Exact initial syy
            syy = -F / a * np.ones_like(x)

            # Exact stress
            stress = [[sxx, sxy], [syx, syy]]

        else:

            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]

            # Exact syy
            c0 = -F / a
            c1 = (-2 * F * (nu_u - nu_s)) / (a * (1 - nu_s))
            syy_sum1 = np.sum(
                (np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.cos(aa_n * x / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            c2 = 2 * F / a
            syy_sum2 = np.sum(
                (np.sin(aa_n) * np.cos(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            syy = c0 + c1 * syy_sum1 + c2 * syy_sum2

            # Exact stress
            stress = [[sxx, sxy], [syx, syy]]

        return stress

    def exact_degree_of_consolidation(self, t: Union[float, int]) -> float:
        """Exact degree of consolidation.
        Parameters:
              t: Time in `s`.
        Returns:
              Exact degree of consolidation for a given time ``t``.
        Notes:
            The degrees of consolidation in the horizontal and vertical axes are identical.
        """
        # Retrieve physical and geometric data
        nu_u = self.undrained_poisson_coefficient()  # [-]
        nu_s = self.poisson_coefficient()  # [-]
        mu_s = self.params["mu_lame"]  # [Pa]
        F = self.params["vertical_load"]  # [N * m^{-1}]
        a, b = self.params["domain_size"]  # ([m], [m])

        # Vertical displacement on the north boundary at time `t`
        uy_b_t = self.exact_vertical_displacement(np.array([b]), t)

        # Initial vertical displacement on the north boundary.
        uy_b_0 = (-F * b * (1 - nu_u)) / (2 * mu_s * a)

        # Vertical displacement on the north boundary at time `infinity`
        uy_b_inf = (-F * b * (1 - nu_s)) / (2 * mu_s * a)

        # Degree of consolidation
        consolidation_degree = (uy_b_t - uy_b_0) / (uy_b_inf - uy_b_0)

        return consolidation_degree

    # -----> Non-primary variables

    def numerical_flux(self) -> np.ndarray:
        """Compute numerical flux.
        Returns:
            Darcy fluxes at the face centers in `m^3 * s^{-1}` with
            ``shape=(sd.num_faces, )``.
        """
        sd = self.mdg.subdomains()[0]
        xf = sd.face_centers[0]
        t = self.time_manager.time
        if t == 0:
            return self.velocity_to_flux(self.exact_velocity(xf, t))
        else:
            flux_ad = self._fluid_flux([sd])
            return flux_ad.evaluate(self.dof_manager).val

    def numerical_traction(self) -> np.ndarray:
        """Compute numerical traction.
        Returns:
            Traction at the face centers in `N` with ``shape=(sd.dim * sd.num_faces, )``.
        """
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        xf = sd.face_centers[0]
        t = self.time_manager.time

        if t == 0:
            return self.ravel(self.stress_to_traction(self.exact_stress(xf, t)))
        else:
            self.reconstruct_stress()
            return data[pp.STATE]["stress"].copy()

    def numerical_consolidation_degree(self) -> tuple[float, float]:
        """
        Compute approximated degree of consolidation.
        Returns:
            Numerical degree of consolidation in the horizontal and vertical directions.
        """
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)

        F = self.params["vertical_load"]  # [N * m^{-1}]
        mu_s = self.params["mu_lame"]  # [Pa]
        nu_s = self.poisson_coefficient()  # [-]
        nu_u = self.undrained_poisson_coefficient()  # [-]
        a, b = self.params["domain_size"]  # [m]

        kw_m = self.mechanics_parameter_key
        disc = data[pp.DISCRETIZATION_MATRICES][kw_m]

        p = data[pp.STATE][self.scalar_variable]
        u = data[pp.STATE][self.displacement_variable]
        t = self.time_manager.time

        if t == 0:
            return 0.0, 0.0

        else:
            bc_vals = data[pp.PARAMETERS][kw_m]["bc_values"]
            bound_u_cell = disc["bound_displacement_cell"]
            bound_u_face = disc["bound_displacement_face"]
            bound_u_pressure = disc["bound_displacement_pressure"]

            trace_u = bound_u_cell * u + bound_u_face * bc_vals + bound_u_pressure * p

            sides = self._domain_boundary_sides(sd)

            trace_ux = trace_u[::2]
            trace_uy = trace_u[1::2]

            ux_a_t = np.max(trace_ux[sides.east])
            ux_a_0 = (F * nu_u) / (2 * mu_s)
            ux_a_inf = (F * nu_s) / (2 * mu_s)
            consol_deg_x = (ux_a_t - ux_a_0) / (ux_a_inf - ux_a_0)

            uy_b_t = np.max(trace_uy[sides.north])
            uy_b_0 = (-F * b * (1 - nu_u)) / (2 * mu_s * a)
            uy_b_inf = (-F * b * (1 - nu_s)) / (2 * mu_s * a)
            consol_deg_y = (uy_b_t - uy_b_0) / (uy_b_inf - uy_b_0)

            return consol_deg_x, consol_deg_y

    # ------> Methods related to non-dimensionalization of variables

    def nondim_t(self, t: Union[float, int, np.ndarray]) -> float:
        """Nondimensionalize time.
        Parameters:
            t: Time in `s`.
        Returns:
            Dimensionless time for the given time ``t``.
        """
        a, _ = self.params["domain_size"]  # [m]
        c_f = self.fluid_diffusivity()  # [m^2 * s^{-1}]

        return (t * c_f) / (a**2)

    def nondim_x(self, x: np.ndarray) -> np.ndarray:
        """Nondimensionalize length in the horizontal axis.
        Parameters:
            x: horizontal length in `m`.
        Returns:
            Dimensionless horizontal length with ``shape=(np, )``.
        """
        a, _ = self.params["domain_size"]  # [m]
        return x / a

    def nondim_y(self, y: np.ndarray) -> np.ndarray:
        """Nondimensionalize length in the vertical axis.
        Parameters:
            y: vertical length in `m`.
        Returns:
            Dimensionless vertical length with ``shape=(np, )``.
        """
        _, b = self.params["domain_size"]  # [m]
        return y / b

    def nondim_p(self, p: np.ndarray) -> np.ndarray:
        """Nondimensionalize pressure.
        Parameters:
            p: Pressure in `Pa`.
        Returns:
            Nondimensional pressure with ``shape=(np, )``.
        """
        a, _ = self.params["domain_size"]  # [m]
        F = self.params["vertical_load"]  # [N * m^{-1}]
        return p / (F * a)

    def nondim_flux(self, qx: np.ndarray) -> np.ndarray:
        """Nondimensionalize horizontal component of the specific discharge vector.
        Parameters:
            qx: Horizontal component of the specific discharge in `m * s^{-1}`.
        Returns:
            Nondimensional horizontal component of the specific discharge with
            ``shape=(np, )``.
        """
        k = self.params["permeability"]  # [m^2]
        F = self.params["vertical_load"]  # [N * m^{-1}]
        mu = self.params["viscosity"]  # [Pa * s]
        a, _ = self.params["domain_size"]  # [m]
        factor = (F * k) / (mu * a**2)  # [m * s^{-1}]
        return qx / factor

    def nondim_stress(self, syy: np.ndarray) -> np.ndarray:
        """Nondimensionalize vertical component of the stress tensor.
        Parameters:
            syy: Vertical component of the stress tensor.
        Returns:
            Nondimensional vertical component of the stress tensor.
        """
        a, _ = self.params["domain_size"]  # [m]
        F = self.params["vertical_load"]  # [N * m^{-1}]
        return syy / (F / a)

    # -----> Utility methods
    def south_cells(self) -> np.ndarray:
        """Get indices of cells that are adjacent to the South boundary"""
        sd = self.mdg.subdomains()[0]
        sides = self._domain_boundary_sides(sd)
        south_idx = np.where(sides.south)[0]
        return sd.signs_and_cells_of_boundary_faces(south_idx)[1]

    def east_cells(self) -> np.ndarray:
        """Get indices of cells that are adjacent to the East boundary"""
        sd = self.mdg.subdomains()[0]
        sides = self._domain_boundary_sides(sd)
        east_idx = np.where(sides.east)[0]
        return sd.signs_and_cells_of_boundary_faces(east_idx)[1]

    def velocity_to_flux(
        self,
        velocity: list[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Convert a velocity field into (integrated normal) fluxes.
        The usual application is to compute (integrated normal) Darcy fluxes form
        specific discharge.
        Parameters:
            velocity: list of arrays in `m * s^{-1}`. Expected shape of each item on
                the list is (sd.num_faces, ).
        Returns:
            Integrated normal fluxes `m^3 * s^{-1}` on the face centers of the grid.
        """
        sd = self.mdg.subdomains()[0]

        # Sanity check on input parameter
        assert velocity[0].size == sd.num_faces
        assert velocity[1].size == sd.num_faces

        flux = velocity[0] * sd.face_normals[0] + velocity[1] * sd.face_normals[1]

        return flux

    def stress_to_traction(
        self,
        stress: list[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray, np.ndarray]:
        """Convert a stress field into (integrated normal) traction forces.
        Parameters:
            stress: list of lists of arrays in `Pa`. Expected shape for each item of
                the list is (sd.num_faces, ).
        Returns:
            List of integrated traction forces `N` on the face centers of the grids.
        """
        sd = self.mdg.subdomains()[0]

        # Sanity check on input parameter
        assert stress[0][0].size == sd.num_faces
        assert stress[0][1].size == sd.num_faces
        assert stress[1][0].size == sd.num_faces
        assert stress[1][1].size == sd.num_faces

        traction_x = (
            stress[0][0] * sd.face_normals[0] + stress[0][1] * sd.face_normals[1]
        )
        traction_y = (
            stress[1][0] * sd.face_normals[0] + stress[1][1] * sd.face_normals[1]
        )

        return [traction_x, traction_y]

    @staticmethod
    def ravel(vector: list[np.ndarray, np.ndarray]) -> np.ndarray:
        """Convert a vector quantity into PorePy format.
        Parameters:
            vector: A vector quantity represented as a list with two items.
        Returns:
            Raveled version of the vector.
        """
        return np.array(vector).ravel("F")

    @staticmethod
    def l2_relative_error(
        sd: pp.Grid,
        true_array: np.ndarray,
        approx_array: np.ndarray,
        is_cc: bool,
        is_scalar: bool,
    ) -> float:
        """Compute the error measured in the discrete (relative) L2-norm.
        The employed norms correspond respectively to equations (75) and (76) for the
        displacement and pressure from
        https://epubs.siam.org/doi/pdf/10.1137/15M1014280.
        Parameters:
            sd: PorePy grid.
            true_array: Exact array, e.g.: pressure, displacement, flux, or traction.
            approx_array: Approximated array, e.g.: pressure, displacement, flux, or
                traction.
            is_cc: True for cell-centered quantities (e.g., pressure and displacement)
                and False for face-centered quantities (e.g., flux and traction).
            is_scalar: True for scalar quantities (e.g., pressure or flux) and False
                for vector quantities (displacement and traction).
        Returns:
            l2_error: discrete L2-error of the quantity of interest.
        """

        if is_cc:
            if is_scalar:
                meas = sd.cell_volumes
            else:
                meas = sd.cell_volumes.repeat(sd.dim)
        else:
            if is_scalar:
                meas = sd.face_areas
            else:
                meas = sd.face_areas.repeat(sd.dim)

        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_array) ** 2))
        l2_error = numerator / denominator

        return l2_error

    # -----> Plotting
    def plot_results(self):
        """Plot results"""

        folder = "out/"
        fname_p = "pressure"
        fname_ux = "horizontal_displacement"
        fname_uy = "vertical_displacement"
        fname_qx = "horizontal_flux"
        fname_syy = "vertical_stress"
        fname_consol = "degree_of_consolidation"
        extension = ".pdf"
        cmap = mcolors.ListedColormap(
            plt.cm.tab20.colors[: len(self.time_manager.schedule)]
        )

        # Non-dimensional pressure
        self._plot_pressure(
            folder=folder, file_name=fname_p, file_extension=extension, color_map=cmap
        )

        # Non-dimensional horizontal displacement
        self._plot_horizontal_displacement(
            folder=folder, file_name=fname_ux, file_extension=extension, color_map=cmap
        )

        # Non-dimensional vertical displacement
        self._plot_vertical_displacement(
            folder=folder, file_name=fname_uy, file_extension=extension, color_map=cmap
        )

        # Non-dimensional horizontal Darcu flux
        self._plot_horizontal_flux(
            folder=folder, file_name=fname_qx, file_extension=extension, color_map=cmap
        )

        # Non-dimensional vertical stress
        self._plot_vertical_stress(
            folder=folder, file_name=fname_syy, file_extension=extension, color_map=cmap
        )

        # Degrees of consolidation
        self._plot_consolidation_degree(
            folder=folder,
            file_name=fname_consol,
            file_extension=extension,
            color_map=cmap,
        )

    def _plot_pressure(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot nondimensional pressure profiles.
        Parameters:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "pressure_profiles".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.
        """
        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]
        south_cells = self.south_cells()

        a, _ = self.params["domain_size"]
        x_ex = np.linspace(0, a, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, sol in enumerate(self.solutions):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_p(self.exact_pressure(x_ex, sol.time)),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xc[south_cells]),
                self.nondim_p(sol.num_pressure[south_cells]),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(sol.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional horizontal distance, $x ~ a^{-1}$", fontsize=13)
        ax.set_ylabel(r"Non-dimensional pressure, $p ~ a ~ F^{-1}$", fontsize=13)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    def _plot_horizontal_displacement(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot non-dimensional horizontal displacement profiles.
        Parameters:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "x_displacement".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.
        """
        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]
        south_cells = self.south_cells()

        a, _ = self.params["domain_size"]
        x_ex = np.linspace(0, a, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, sol in enumerate(self.solutions):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_x(self.exact_horizontal_displacement(x_ex, sol.time)),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xc[south_cells]),
                self.nondim_x(sol.num_displacement_x[south_cells]),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(sol.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional horizontal distance, $x ~ a^{-1}$", fontsize=13)
        ax.set_ylabel(
            r"Non-dimensional horizontal displacement, $u_x ~ a^{-1}$", fontsize=13
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    def _plot_vertical_displacement(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot non-dimensional vertical displacement profiles.
        Parameters:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "y_displacement".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.
        """
        sd = self.mdg.subdomains()[0]
        yc = sd.cell_centers[1]
        east_cells = self.east_cells()

        _, b = self.params["domain_size"]
        y_ex = np.linspace(0, b, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, sol in enumerate(self.solutions):
            ax.plot(
                self.nondim_y(y_ex),
                self.nondim_y(self.exact_vertical_displacement(y_ex, sol.time)),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_y(yc[east_cells]),
                self.nondim_y(sol.num_displacement_y[east_cells]),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(sol.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional vertical distance, $y ~ b^{-1}$", fontsize=13)
        ax.set_ylabel(
            r"Non-dimensional vertical displacement, $u_y ~ b^{-1}$", fontsize=13
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    def _plot_horizontal_flux(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot non-dimensional horizontal Darcy flux profiles.
        Parameters:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "flux".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.
        """
        sd = self.mdg.subdomains()[0]
        xf = sd.face_centers[0]
        nx = sd.face_normals[0]
        sides = self._domain_boundary_sides(sd)
        south_cells = self.south_cells()
        faces_of_south_cells = sps.find(sd.cell_faces.T[south_cells])[1]
        south_faces = np.where(sides.south)[0]
        int_faces_of_south_cells = np.setdiff1d(faces_of_south_cells, south_faces)

        a, _ = self.params["domain_size"]
        x_ex = np.linspace(0, a, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, sol in enumerate(self.solutions):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_flux(self.exact_velocity(x_ex, sol.time)[0]),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xf[int_faces_of_south_cells]),
                self.nondim_flux(
                    sol.num_flux[int_faces_of_south_cells]
                    / nx[int_faces_of_south_cells]
                ),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(sol.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional horizontal distance, $x ~ a^{-1}$", fontsize=13)
        ax.set_ylabel(
            r"Non-dimensional horizontal flux, "
            r"$q_x ~ \mu_f ~ a^2 ~ F^{-1} ~ k^{-1}$",
            fontsize=13,
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    def _plot_vertical_stress(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot non-dimensional vertical stress profiles.
        Parameters:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "sigma_yy".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.
        """
        sd = self.mdg.subdomains()[0]
        xf = sd.face_centers[0]
        sides = self._domain_boundary_sides(sd, 1e-6)
        south_faces = sides.south
        ny = sd.face_normals[1]

        a, b = self.params["domain_size"]
        x_ex = np.linspace(0, a, 400)

        # Vertical stress plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, sol in enumerate(self.solutions):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_stress(self.exact_stress(x_ex, sol.time)[1][1]),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xf[south_faces]),
                self.nondim_stress(
                    sol.num_traction[1 :: sd.dim][south_faces] / ny[south_faces]
                ),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${round(self.nondim_t(sol.time), 5)}",
            )
        ax.set_xlabel(
            r"Non-dimensional horizontal distance, $ x ~ a^{-1}$", fontsize=13
        )
        ax.set_ylabel(
            r"Non-dimensional vertical poroelastic stress,"
            r" $\sigma_{yy} ~ a ~ F^{-1}$",
            fontsize=13,
        )
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    def _plot_consolidation_degree(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ):
        """Plot degree of consolidation as a function of time."""

        # Retrieve data
        a, _ = self.params["domain_size"]  # [m]
        c_f = self.fluid_diffusivity()  # [m^2 * s^{-1}]

        # Generate exact consolidation times
        tau_0 = 1e-3
        tau_1 = 1e-2
        tau_2 = 1e0
        tau_3 = 1e1

        t = lambda tau: tau * a**2 / c_f

        interval_0 = np.linspace(t(tau_0), t(tau_1), 100)
        interval_1 = np.linspace(t(tau_1), t(tau_2), 100)
        interval_2 = np.linspace(t(tau_2), t(tau_3), 100)
        ex_times = np.concatenate((interval_0, interval_1))
        ex_times = np.concatenate((ex_times, interval_2))

        ex_consol_degree = np.array(
            [self.exact_degree_of_consolidation(t) for t in ex_times]
        )

        # Numerical consolidation degrees
        num_times = self.time_manager.schedule
        num_consol_degree_x = np.array(
            [sol.num_consol_degree[0] for sol in self.solutions]
        )
        num_consol_degree_y = np.array(
            [sol.num_consol_degree[1] for sol in self.solutions]
        )

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.semilogx(
            self.nondim_t(ex_times),
            ex_consol_degree,
            alpha=0.5,
            color="black",
            linewidth=2,
            label="Exact",
        )
        ax.semilogx(
            self.nondim_t(num_times),
            num_consol_degree_x,
            marker="s",
            linewidth=0,
            markerfacecolor="none",
            markeredgewidth=2,
            color="blue",
            markersize=10,
            label="Horizontal MPFA/MPSA-FV",
        )
        ax.semilogx(
            self.nondim_t(num_times),
            num_consol_degree_y,
            marker="+",
            markeredgewidth=2,
            linewidth=0,
            color="red",
            markersize=10,
            label="Vertical MPFA/MPSA-FV",
        )
        ax.set_xlabel(r"Non-dimensional time, $t ~ c_f ~ a^{-2}$", fontsize=13)
        ax.set_ylabel(
            r"Degree of consolidation," r" $U(t)$",
            fontsize=13,
        )
        ax.grid()
        ax.legend(fontsize=12)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()
