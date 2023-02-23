"""
This module contains the implementation of Mandel's consolidation problem. The problem
is discretized using MPFA/MPSA-FV in space and backward Euler in time. For further
details on Mandel's problem, see [1 – 3]. For the implementation details, see [4].

References:

    - [1] Mandel, J.: Consolidation des sols (étude mathématique). Geotechnique.
      3(7), 287–299 (1953).

    - [2] Cheng, A.H.-D., Detournay, E.: A direct boundary element method for plane
     strain poroelasticity. Int. J. Numer. Anal. Methods Geomech. 12(5), 551–572 (1988).

    - [3] Mikelić, A., Wang, B., & Wheeler, M. F. (2014). Numerical convergence study of
      iterative coupling for coupled flow and geomechanics. Computational Geosciences,
      18(3), 325-341.

    - [4] Keilegavlen, E., Berge, R., Fumagalli, A. et al. PorePy: an open-source
      software for simulation of multiphysics processes in fractured porous media.
      Comput Geosci 25, 243–265 (2021).

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import matplotlib.colors as mcolors  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sps

import porepy as pp
import porepy.models.fluid_mass_balance as mass
import porepy.models.poromechanics as poromechanics
from porepy.applications.building_blocks.derived_models.biot import BiotPoromechanics
from porepy.applications.building_blocks.verification_utils import VerificationUtils
from porepy.viz.data_saving_model_mixin import VerificationDataSaving

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
    "density": 1e3,  # [kg * m^-3]
    "viscosity": 1e-3,  # [Pa * s]
}


# -----> Data-saving
@dataclass
class MandelSaveData:
    """Class to store relevant data from the verification setup."""

    approx_consolidation_degree: tuple[number, number]
    """Numerical degrees of consolidation in the horizontal and vertical directions."""

    approx_displacement: np.ndarray
    """Numerical displacement."""

    approx_flux: np.ndarray
    """Numerical Darcy flux."""

    approx_force: np.ndarray
    """Numerical poroelastic force."""

    approx_pressure: np.ndarray
    """Numerical pressure."""

    error_consolidation_degree: tuple[number, number]
    """Absolute error for the degrees of consolidation in the horizontal and vertical
    directions.

    """

    error_displacement: number
    """L2-discrete relative error for the displacement."""

    error_flux: number
    """L2-discrete relative error for the Darcy flux."""

    error_force: number
    """L2-discrete relative error for the poroelastic force."""

    error_pressure: number
    """L2-discrete relative error for the pressure."""

    exact_consolidation_degree: number
    """Exact degree of consolidation."""

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


class MandelDataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`~porepy.models.fluid_mass_balance.DarcysLaw`.

    """

    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """

    exact_sol: MandelExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of storing and scaling numerical values
    representing fluid-related quantities. Normally, this is set by an instance of
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    numerical_consolidation_degree: Callable[[], tuple[number, number]]
    """Numerical degree of consolidation in the horizontal and vertical directions."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    relative_l2_error: Callable
    """Method for computing the discrete relative L2-error. Normally provided by a
    mixin instance of :class:`~porepy.applications.building_blocks.
    verification_utils.VerificationUtils`.

    """

    stress: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the (integrated) poroelastic stress in the form of an Ad
    operator. Usually provided by the mixin class
    :class:`porepy.models.poromechanics.ConstitutiveLawsPoromechanics`.

    """

    def collect_data(self) -> MandelSaveData:
        """Collect the data from the verification setup.

        Returns:
            MandelSaveData object containing the results of the verification for the
            current time.

        """

        # Retrieve information from setup
        mdg: pp.MixedDimensionalGrid = self.mdg
        sd: pp.Grid = mdg.subdomains()[0]
        t: number = self.time_manager.time

        # Collect data
        exact_pressure = self.exact_sol.pressure(sd, t)
        pressure_ad = self.pressure([sd])
        approx_pressure = pressure_ad.evaluate(self.equation_system).val
        error_pressure = self.relative_l2_error(
            grid=sd,
            true_array=exact_pressure,
            approx_array=approx_pressure,
            is_scalar=True,
            is_cc=True,
        )

        exact_displacement = self.exact_sol.displacement(sd, t)
        displacement_ad = self.displacement([sd])
        approx_displacement = displacement_ad.evaluate(self.equation_system).val
        error_displacement = self.relative_l2_error(
            grid=sd,
            true_array=exact_displacement,
            approx_array=approx_displacement,
            is_scalar=False,
            is_cc=True,
        )

        exact_flux = self.exact_sol.flux(sd, t)
        flux_ad = self.darcy_flux([sd])
        mobility = 1 / self.fluid.viscosity()
        approx_flux = mobility * flux_ad.evaluate(self.equation_system).val
        error_flux = self.relative_l2_error(
            grid=sd,
            true_array=exact_flux,
            approx_array=approx_flux,
            is_scalar=True,
            is_cc=False,
        )

        exact_force = self.exact_sol.poroelastic_force(sd, t)
        force_ad = self.stress([sd])
        approx_force = force_ad.evaluate(self.equation_system).val
        error_force = self.relative_l2_error(
            grid=sd,
            true_array=exact_force,
            approx_array=approx_force,
            is_scalar=False,
            is_cc=False,
        )

        exact_consolidation_degree = self.exact_sol.degree_of_consolidation(t)
        approx_consolidation_degree = self.numerical_consolidation_degree()
        error_consolidation_degree_x = np.abs(
            approx_consolidation_degree[0] - exact_consolidation_degree
        )
        error_consolidation_degree_y = np.abs(
            approx_consolidation_degree[1] - exact_consolidation_degree
        )
        error_consolidation_degree = (
            float(error_consolidation_degree_x),
            float(error_consolidation_degree_y),
        )

        # Store collected data in data class
        collected_data = MandelSaveData(
            approx_consolidation_degree=approx_consolidation_degree,
            approx_displacement=approx_displacement,
            approx_flux=approx_flux,
            approx_force=approx_force,
            approx_pressure=approx_pressure,
            error_consolidation_degree=error_consolidation_degree,
            error_displacement=error_displacement,
            error_flux=error_flux,
            error_force=error_force,
            error_pressure=error_pressure,
            exact_consolidation_degree=exact_consolidation_degree,
            exact_displacement=exact_displacement,
            exact_flux=exact_flux,
            exact_force=exact_force,
            exact_pressure=exact_pressure,
            time=t,
        )

        return collected_data


# -----> Exact solution
class MandelExactSolution:
    """Exact solutions to Mandel's problem taken from [2]."""

    def __init__(self, setup):
        """Constructor of the class."""
        self.setup = setup

        # For convenience, store the approximated roots
        self.roots: np.ndarray = self.approximate_roots()

    def approximate_roots(self) -> np.ndarray:
        """Approximate roots of infinite series using the bisection method.

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
        nu_s = self.setup.poisson_coefficient()  # [-]
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]

        # Define algebraic function
        def f(x):
            y = np.tan(x) - ((1 - nu_s) / (nu_u - nu_s)) * x
            return y

        n_series = self.setup.params.get("number_of_roots", 200)
        a_n = np.zeros(n_series)  # initialize roots
        x0 = 0.0  # initial point
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

    # ---> Pressure
    def pressure_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact fluid pressure in scaled [Pa].

        Parameters:
            x: Points in the horizontal axis in scaled [m].
            t: Time in scaled [s].

        Returns:
            Exact pressure solution with ``shape=(num_points, )``.

        """

        # Retrieve data
        F = self.setup.vertical_load()  # scaled [N * m^-1]
        B = self.setup.skempton_coefficient()  # [-]
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]
        c_f = self.setup.fluid_diffusivity()  # scaled [m^2 * s^-1]
        a = self.setup.domain.bounding_box["xmax"]  # scaled [m]

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Compute exact pressure
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

    def pressure(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact pressure in scaled [Pa] at the cell centers.

        Parameters:
            sd: Grid.
            t: Time in scaled [s].

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact pressures at
            the cell centers at the given time ``t``.

        """
        return self.pressure_profile(sd.cell_centers[0], t)

    # ---> Displacements
    def horizontal_displacement_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact horizontal displacement in scaled [m].

        Parameters:
            x: Points in the horizontal axis in scaled [m].
            t: Time in scaled [s].

        Returns:
            Exact horizontal displacement with ``shape=(num_points, )``.

        """

        # Retrieve data
        F = self.setup.vertical_load()  # scaled [N * m^-1]
        nu_s = self.setup.poisson_coefficient()  # [-]
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]
        mu_s = self.setup.solid.shear_modulus()  # scaled [Pa]
        c_f = self.setup.fluid_diffusivity()  # scaled [m^2 * s^-1]
        a = self.setup.domain.bounding_box["xmax"]  # scaled [m]

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Compute horizontal displacement
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
        """Exact vertical displacement in scaled [m].

        Parameters:
            y: Points in the horizontal axis in scaled [m].
            t: Time in scaled [s].

        Returns:
            Exact vertical displacement with ``shape=(num_points, )``.

        """

        # Retrieve data
        F = self.setup.vertical_load()  # scaled [N * m^-1]
        nu_s = self.setup.poisson_coefficient()  # [-]
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]
        mu_s = self.setup.solid.shear_modulus()  # scaled [Pa]
        c_f = self.setup.fluid_diffusivity()  # scaled [m^2 * s^-1]
        a = self.setup.domain.bounding_box["xmax"]  # scaled [m]

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Compute exact vertical displacement
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

    def displacement(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact displacement in scaled [m] at the cell centers.

        Parameters:
            sd: Grid.
            t: Time in scaled [s].

        Returns:
            Array of ``shape=(sd.num_cells * 2, )`` containing the exact displacement at
            the cell centers at the given time ``t``. The returned array is given in
            flattened vector format.

        """
        ux = self.horizontal_displacement_profile(sd.cell_centers[0], t)
        uy = self.vertical_displacement_profile(sd.cell_centers[1], t)
        return np.stack((ux, uy)).ravel("F")

    # ---> Flux
    def horizontal_velocity_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact horizontal specific discharge in scaled [m * s^-1].

        Note that for Mandel's problem, the vertical specific discharge is zero.

        Parameters:
            x: Points in the horizontal axis in scaled [m].
            t: Time in scaled [s].

        Returns:
            Exact specific discharge for the given time ``t``. The returned array has
             ``shape=(num_points, )``.

        """
        # Retrieve physical data
        F = self.setup.vertical_load()  # scaled [N * m^-1]
        B = self.setup.skempton_coefficient()  # [-]
        k = self.setup.solid.permeability()  # scaled [m^2]
        mu_f = self.setup.fluid.viscosity()  # scaled [Pa * s]
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]
        c_f = self.setup.fluid_diffusivity()  # scaled [m^2 * s^-1]
        a = self.setup.domain.bounding_box["xmax"]  # scaled [m]

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Compute exact horizontal specific discharge
        if t == 0:
            qx = np.zeros_like(x)  # zero initial discharge
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

    def flux(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact Darcy flux in scaled [m^3 * s^-1] at the face centers.

        Parameters:
            sd: Grid.
            t: Time in scaled [s].

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact Darcy flux at
            the face centers at the given time ``t``.

        """
        q_x = self.horizontal_velocity_profile(sd.face_centers[0], t)
        n_x = sd.face_normals[0]
        return q_x * n_x

    # ---> Force
    def vertical_stress_profile(self, x: np.ndarray, t: number) -> np.ndarray:
        """Exact vertical stress in scaled [Pa].

        Note that for Mandel's problem, all the other components of the stress tensor
        are zero.

        Parameters:
            x: Points in the horizontal axis in scaled [m].
            t: Time in scaled [s].

        Returns:
            Exact vertical component of the symmetric stress tensor with
            ``shape=(num_points, )``.

        """
        # Retrieve physical data
        F = self.setup.vertical_load()  # scaled [N * m^-1]
        nu_s = self.setup.poisson_coefficient()  # [-]
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]
        c_f = self.setup.fluid_diffusivity()  # scaled [m^2 * s^-1]
        a = self.setup.domain.bounding_box["xmax"]  # scaled [m]

        # Retrieve roots
        aa_n = self.roots[:, np.newaxis]

        # Compute exact vertical component of the stress tensor
        if t == 0:  # vertical stress at t = 0 has a different expression
            s_yy = -F / a * np.ones_like(x)
        else:
            c0 = -F / a
            c1 = (-2 * F * (nu_u - nu_s)) / (a * (1 - nu_s))
            c2 = 2 * F / a
            s_yy_sum1 = np.sum(
                (np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.cos(aa_n * x / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            s_yy_sum2 = np.sum(
                (np.sin(aa_n) * np.cos(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            s_yy = c0 + c1 * s_yy_sum1 + c2 * s_yy_sum2

        return s_yy

    def poroelastic_force(self, sd: pp.Grid, t: number) -> np.ndarray:
        """Evaluate exact poroelastic force in scaled [N] at the face centers.

        Parameters:
            sd: Grid.
            t: Time in scaled [s].

        Returns:
            Array of ``shape=(sd.num_faces * 2, )`` containing the exact poroelastic
            force at the face centers at the given time ``t``.

        """
        s_yy = self.vertical_stress_profile(sd.face_centers[0], t)
        n_y = sd.face_normals[1]
        force_y = s_yy * n_y
        force_x = np.zeros_like(force_y)
        return np.stack((force_x, force_y)).ravel("F")

    # ---> Degree of consolidation
    def degree_of_consolidation(self, t: number) -> number:
        """Exact degree of consolidation [-].

        Note that for Mandel's problem, the degrees of consolidation in the
        horizontal and vertical axes are identical.

        Parameters:
              t: Time in scaled [s].

        Returns:
              Exact degree of consolidation for a given time ``t``.

        """
        # Retrieve physical and geometric data
        nu_u = self.setup.undrained_poisson_coefficient()  # [-]
        nu_s = self.setup.poisson_coefficient()  # [-]
        mu_s = self.setup.solid.shear_modulus()  # scaled [Pa]
        F = self.setup.vertical_load()  # scaled [N * m^-1]
        a = self.setup.domain.bounding_box["xmax"]  # scaled [m]
        b = self.setup.domain.bounding_box["ymax"]  # scaled [m]

        # Vertical displacement on the north boundary at time `t`
        uy_b_t = self.vertical_displacement_profile(np.array([b]), t)

        # Initial vertical displacement on the north boundary
        uy_b_0 = (-F * b * (1 - nu_u)) / (2 * mu_s * a)

        # Vertical displacement on the north boundary at time `infinity`
        uy_b_inf = (-F * b * (1 - nu_s)) / (2 * mu_s * a)

        # Degree of consolidation
        consolidation_degree = (uy_b_t - uy_b_0) / (uy_b_inf - uy_b_0)

        return consolidation_degree


# -----> Utilities
class MandelUtils(VerificationUtils):
    """Mixin class that provides useful utility methods for the verification setup."""

    domain: pp.Domain
    """Domain specification. Set by an instance of :class:`~MandelGeometry`."""

    domain_boundary_sides: Callable[[pp.Grid], pp.domain.DomainSides]
    """Boundary sides of the domain. Defined by a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    exact_sol: MandelExactSolution
    """Exact solution object. Normally set by an instance of
    :class:`~MandelExactSolution`.

    """

    results: list[MandelSaveData]
    """List of :class:`MandelSaveData` objects containing the results of the
    verification.

    """

    vertical_load: Callable
    """Method that retrieves the applied vertical load from the model parameters and
    applies the proper scaling. Provided by an instance of
    :class:~`MandelBoundaryConditionsMechanicsTimeDependent`.

    """

    # -----> Derived physical constants
    def bulk_modulus(self) -> number:
        """Set bulk modulus in scaled [Pa].

        Returns:
            Bulk modulus.

        """
        mu_s = self.solid.shear_modulus()  # scaled [Pa]
        lambda_s = self.solid.lame_lambda()  # scaled [Pa]
        return (2 / 3) * mu_s + lambda_s

    def poisson_coefficient(self) -> number:
        """Set Poisson coefficient [-]

        Returns:
            Poisson coefficient.

        """
        mu_s = self.solid.shear_modulus()
        K_s = self.bulk_modulus()
        return (3 * K_s - 2 * mu_s) / (2 * (3 * K_s + mu_s))

    def undrained_bulk_modulus(self) -> number:
        """Set undrained bulk modulus in scaled [Pa].

        Returns:
            Undrained bulk modulus.

        """
        alpha_biot = self.solid.biot_coefficient()  # [-]
        K_s = self.bulk_modulus()  # scaled [Pa]
        S_epsilon = self.solid.specific_storage()  # scaled [Pa^-1]
        return K_s + (alpha_biot**2) / S_epsilon

    def skempton_coefficient(self) -> number:
        """Set Skempton's coefficient [-].

        Returns:
            Skempton's coefficent.

        """
        alpha_biot = self.solid.biot_coefficient()  # [-]
        K_u = self.undrained_bulk_modulus()  # scaled [Pa]
        S_epsilon = self.solid.specific_storage()  # scaled [Pa^-1]
        return alpha_biot / (S_epsilon * K_u)

    def undrained_poisson_coefficient(self) -> float:
        """Set Poisson coefficient under undrained conditions [-].

        Returns:
            Undrained Poisson coefficient.

        Notes:
            Expression taken from https://doi.org/10.1016/j.camwa.2018.09.005.

        """
        nu_s = self.poisson_coefficient()  # [-]
        B = self.skempton_coefficient()  # [-]
        return (3 * nu_s + B * (1 - 2 * nu_s)) / (3 - B * (1 - 2 * nu_s))

    def fluid_diffusivity(self) -> number:
        """Set fluid diffusivity scaled [m^2 * s^-1].

        Returns:
            Fluid diffusivity.

        """
        k_s = self.solid.permeability()  # scaled [m^2]
        B = self.skempton_coefficient()  # [-]
        mu_s = self.solid.shear_modulus()  # scaled [Pa]
        nu_s = self.poisson_coefficient()  # [-]
        nu_u = self.undrained_poisson_coefficient()  # [-]
        mu_f = self.fluid.viscosity()  # scaled [Pa * s]
        c_f = (2 * k_s * (B**2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / (
            9 * mu_f * (1 - nu_u) * (nu_u - nu_s)
        )
        return c_f

    # -----> Non-dimensionalization methods
    def nondim_time(self, t: Union[number, np.ndarray]) -> Union[number, np.ndarray]:
        """Non-dimensionalize time.

        Parameters:
            t: Time in scaled [s].

        Returns:
            Dimensionless time for the given time ``t``.

        """
        a = self.domain.bounding_box["xmax"]  # scaled [m]
        c_f = self.fluid_diffusivity()  # scaled [m^2 * s^-1]
        return (t * c_f) / (a**2)

    def nondim_x(self, x: np.ndarray) -> np.ndarray:
        """Nondimensionalize length in the horizontal direction.

        Parameters:
            x: horizontal length in scaled [m].

        Returns:
            Dimensionless horizontal length with ``shape=(num_points, )``.

        """
        a = self.domain.bounding_box["xmax"]  # scaled [m]
        return x / a

    def nondim_y(self, y: np.ndarray) -> np.ndarray:
        """Non-dimensionalize length in the vertical direction.

        Parameters:
            y: vertical length in scaled [m].

        Returns:
            Dimensionless vertical length with ``shape=(num_points, )``.

        """
        b = self.domain.bounding_box["ymax"]  # scaled [m]
        return y / b

    def nondim_pressure(self, p: np.ndarray) -> np.ndarray:
        """Non-dimensionalize pressure.

        Parameters:
            p: Fluid pressure in scaled [Pa].

        Returns:
            Dimensionless pressure with ``shape=(num_points, )``.

        Note:
            This method is also valid to non-dimensionalize stress.

        """
        F = self.vertical_load()  # scaled [N * m^-1]
        a = self.domain.bounding_box["xmax"]  # scaled [m]
        return p / (F * a)

    def nondim_flux(self, q_x: np.ndarray) -> np.ndarray:
        """Non-dimensionalize horizontal component of the specific discharge vector.

        Parameters:
            q_x: Horizontal component of the specific discharge in scaled [m * s^-1].

        Returns:
            Dimensionless horizontal component of the specific discharge with
            ``shape=(num_points, )``.

        """
        k = self.solid.permeability()  # scaled [m^2]
        F = self.vertical_load()  # scaled [N * m^-1]
        mu = self.fluid.viscosity()  # scaled [Pa * s]
        a = self.domain.bounding_box["xmax"]  # scaled [m]
        factor = (F * k) / (mu * a**2)  # scaled [m * s^-1]
        return q_x / factor

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

    def numerical_consolidation_degree(self) -> tuple[number, number]:
        """Numerical consolidation degree [-].

        Returns:
            Numerical degree of consolidation in the horizontal and vertical directions.

        """
        sd = self.mdg.subdomains()[0]
        sides = self.domain_boundary_sides(sd)
        a = self.domain.bounding_box["xmax"]  # scaled [m]
        b = self.domain.bounding_box["ymax"]  # scaled [m]

        F = self.vertical_load()  # scaled [N * m^-1]
        mu_s = self.solid.shear_modulus()  # scaled [Pa]
        nu_s = self.poisson_coefficient()  # [-]
        nu_u = self.undrained_poisson_coefficient()  # [-]

        t = self.time_manager.time  # scaled [s]

        # Retrieve face displacement
        u_faces = self.face_displacement(sd)

        if t == 0:  # soil is initially unconsolidated
            consol_deg_x, consol_deg_y = 0, 0
        else:
            # Consolidation degree in the horizontal direction
            ux_a_t = np.max(u_faces[::2][sides.east])
            ux_a_0 = (F * nu_u) / (2 * mu_s)
            ux_a_inf = (F * nu_s) / (2 * mu_s)
            consol_deg_x = (ux_a_t - ux_a_0) / (ux_a_inf - ux_a_0)

            # Consolidation degree in the vertical direction
            uy_b_t = np.max(u_faces[1::2][sides.north])
            uy_b_0 = (-F * b * (1 - nu_u)) / (2 * mu_s * a)
            uy_b_inf = (-F * b * (1 - nu_s)) / (2 * mu_s * a)
            consol_deg_y = (uy_b_t - uy_b_0) / (uy_b_inf - uy_b_0)

        return consol_deg_x, consol_deg_y

    # -----> Plotting methods
    def plot_results(self):
        """Plot results."""
        num_lines = len(self.time_manager.schedule) - 1
        cmap = mcolors.ListedColormap(plt.cm.tab20.colors[:num_lines])

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

        a = self.domain.bounding_box["xmax"]  # scaled [m]
        x_ex = np.linspace(0, a, 400)

        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, result in enumerate(self.results):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_pressure(
                    self.exact_sol.pressure_profile(x_ex, result.time)
                ),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xc[south_cells]),
                self.nondim_pressure(result.approx_pressure[south_cells]),
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
                label=rf"$\tau=${np.round(self.nondim_time(result.time), 5)}",
            )
        ax.set_xlabel(r"Non-dimensional horizontal distance, $x ~ a^{-1}$", fontsize=13)
        ax.set_ylabel(r"Non-dimensional pressure, $p ~ a ~ F^{-1}$", fontsize=13)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.grid()
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

        a = self.domain.bounding_box["xmax"]  # scaled [m]
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
                label=rf"$\tau=${np.round(self.nondim_time(result.time), 5)}",
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

        b = self.domain.bounding_box["ymax"]  # [m]
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
                label=rf"$\tau=${np.round(self.nondim_time(result.time), 5)}",
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

        a = self.domain.bounding_box["xmax"]  # scaled [m]
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
                label=rf"$\tau=${np.round(self.nondim_time(result.time), 5)}",
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

        a = self.domain.bounding_box["xmax"]  # scaled [m]
        x_ex = np.linspace(0, a, 400)

        # Vertical stress plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, result in enumerate(self.results):
            ax.plot(
                self.nondim_x(x_ex),
                self.nondim_pressure(
                    self.exact_sol.vertical_stress_profile(x_ex, result.time)
                ),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.nondim_x(xf[south_faces]),
                self.nondim_pressure(
                    result.approx_force[1 :: sd.dim][south_faces] / ny[south_faces]
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
                label=rf"$\tau=${np.round(self.nondim_time(result.time), 5)}",
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
        m = 1 / self.units.m
        a, _ = self.params.get("domain_size", (100, 10))  # [m]
        a *= m
        c_f = self.fluid_diffusivity()  # [m^2 * s^-1]

        # Generate exact consolidation times
        tau_0 = 1e-4
        tau_1 = 1e-3
        tau_2 = 1e-2
        tau_3 = 1e0
        tau_4 = 1e1

        # Physical time as a function of dimensionless time
        t = lambda tau: tau * a**2 / c_f

        # Get array of physical times
        interval_0 = np.linspace(t(tau_0), t(tau_1), 100)
        interval_1 = np.linspace(t(tau_1), t(tau_2), 100)
        interval_2 = np.linspace(t(tau_2), t(tau_3), 100)
        interval_3 = np.linspace(t(tau_3), t(tau_4), 100)
        ex_times = np.concatenate((interval_0, interval_1))
        ex_times = np.concatenate((ex_times, interval_2))
        ex_times = np.concatenate((ex_times, interval_3))

        # Exact consolidation degree
        ex_consol_degree = np.array(
            [self.exact_sol.degree_of_consolidation(t) for t in ex_times]
        )

        # Numerical consolidation degrees
        times = self.time_manager.schedule[1:]
        approx_consol_degree_x = np.array(
            [result.approx_consolidation_degree[0] for result in self.results]
        )
        approx_consol_degree_y = np.array(
            [result.approx_consolidation_degree[1] for result in self.results]
        )

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.semilogx(
            self.nondim_time(ex_times),
            ex_consol_degree,
            alpha=0.5,
            color="black",
            linewidth=2,
            label="Exact",
        )
        ax.semilogx(
            self.nondim_time(times),
            approx_consol_degree_x,
            marker="s",
            linewidth=0,
            markerfacecolor="none",
            markeredgewidth=2,
            color="blue",
            markersize=10,
            label="Horizontal MPFA/MPSA-FV",
        )
        ax.semilogx(
            self.nondim_time(times),
            approx_consol_degree_y,
            marker="+",
            markeredgewidth=2,
            linewidth=0,
            color="red",
            markersize=10,
            label="Vertical MPFA/MPSA-FV",
        )
        ax.set_xlabel(r"Non-dimensional time, $t ~ c_f ~ a^{-2}$", fontsize=13)
        ax.set_ylabel(
            "Degree of consolidation," r" $U(t)$",
            fontsize=13,
        )
        ax.grid()
        ax.legend(fontsize=12)
        plt.show()


# -----> Geometry
class MandelGeometry(pp.ModelGeometry):
    """Class for setting up the rectangular geometry."""

    params: dict
    """Simulation model parameters."""

    fracture_network: pp.FractureNetwork2d
    """Two-dimensional fracture network object. Set by a mixin instance of
    :class:`~MandelGeometry`.

    """

    def set_fracture_network(self) -> None:
        """Set fracture network. Unit square with no fractures."""
        ls = 1 / self.units.m  # length scaling
        a, b = self.params.get("domain_size", (100, 10))  # [m]
        domain = pp.Domain({"xmin": 0.0, "xmax": a * ls, "ymin": 0.0, "ymax": b * ls})
        self.fracture_network = pp.FractureNetwork2d(None, domain)

    def mesh_arguments(self) -> dict:
        """Set mesh arguments."""
        ls = 1 / self.units.m  # length scaling
        default_mesh_arguments = {
            "mesh_size_frac": 2 * ls,
            "mesh_size_bound": 2 * ls,
        }
        return self.params.get("mesh_arguments", default_mesh_arguments)

    def set_md_grid(self) -> None:
        """Set mixed-dimensional grid."""
        self.mdg = self.fracture_network.mesh(self.mesh_arguments())
        domain = self.fracture_network.domain
        if domain is not None and domain.is_boxed:
            self.domain: pp.Domain = domain


# -----> Boundary conditions
class MandelBoundaryConditionsMechanicsTimeDependent(
    poromechanics.BoundaryConditionsMechanicsTimeDependent,
):
    domain_boundary_sides: Callable[[pp.Grid], pp.domain.DomainSides]
    """Boundary sides of the domain. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    exact_sol: MandelExactSolution
    """Exact solution object."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    params: dict
    """Parameter dictionary of the verification setup."""

    stress_keyword: str
    """Keyword for accessing the parameters of the mechanical subproblem."""

    time_manager: pp.TimeManager
    """Time manager. Normally set by an instance of a subclass of
    :class:`porepy.models.solution_strategy.SolutionStrategy`.

    """

    units: pp.Units
    """Units object, containing the scaling of base magnitudes."""

    def vertical_load(self):
        """Retrieve and scale applied force.

        Returns:
            Applied vertical load on the North boundary in scaled [N * m^-1].

        """
        N = 1 / self.units.N  # force scaling
        m = 1 / self.units.m  # length scaling
        applied_force = self.params.get("vertical_load", 6e8)  # [N * m^-1]
        return applied_force * (N / m)

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions.

        Rollers on all sides except on the East side which is Neumann.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition representation.

        """
        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        bc = super().bc_type_mechanics(sd=sd)

        # Get boundary sides, retrieve data dict, and bc object
        sides = self.domain_boundary_sides(sd)

        # East side: Neumann
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

    def time_dependent_bc_values_mechanics(
        self, subdomains: list[pp.Grid]
    ) -> np.ndarray:
        """Boundary values for mechanics.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values. Only non-zero values are the ones associated to
            the North side of the domain.

        """
        bc_vals = super().time_dependent_bc_values_mechanics(subdomains)

        sd = subdomains[0]
        sides = self.domain_boundary_sides(sd)
        yf_north = sd.face_centers[1][sides.north]

        t = self.time_manager.time  # scaled [s]
        uy_north_bc = self.exact_sol.vertical_displacement_profile(yf_north, t)
        bc_vals[1::2][sides.north] = uy_north_bc

        return bc_vals


class MandelBoundaryConditionsSinglePhaseFlow(mass.BoundaryConditionsSinglePhaseFlow):

    domain_boundary_sides: Callable[[pp.Grid], pp.domain.DomainSides]
    """Boundary sides of the domain. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        All sides are set as Neumann, except the East side, which is Dirichlet.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Scalar boundary condition representation.

        """
        return pp.BoundaryCondition(sd, self.domain_boundary_sides(sd).east, "dir")


class MandelPoromechanicsBoundaryConditions(
    MandelBoundaryConditionsSinglePhaseFlow,
    MandelBoundaryConditionsMechanicsTimeDependent,
):
    """Mixer class for poromechanics boundary conditions."""


# -----> Solution strategy
class MandelSolutionStrategy(poromechanics.SolutionStrategyPoromechanics):
    """Solution strategy for Mandel's problem."""

    exact_sol: MandelExactSolution
    """Exact solution object."""

    plot_results: Callable[[], None]
    """Method that plots pressure, displacement, flux, force, and degree of
    consolidation in non-dimensional forms.

    """

    results: list[MandelSaveData]
    """List of :class:~`MandelSaveData` objects, containing the results of the
    verification.

    """

    def __init__(self, params: dict) -> None:
        """Constructor of the class.

        Parameters:
            params: Parameters of the verification setup.

        """
        super().__init__(params)

        self.exact_sol: MandelExactSolution
        """Exact solution object."""

        self.results: list[MandelSaveData] = []
        """List of stored results from the verification."""

    def set_materials(self):
        """Set material parameters.

        Add exact solution object to the simulation model after materials have been set.

        """
        super().set_materials()
        self.exact_sol = MandelExactSolution(self)

        # Biot's coefficient must be one
        assert self.solid.biot_coefficient() == 1

    def initial_condition(self) -> None:
        """Set initial conditions.

        Initial conditions are given by Eqs. (41) - (43) from [3].

        """
        super().initial_condition()

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        p_name = self.pressure_variable
        u_name = self.displacement_variable

        # Set initial pressure
        data[pp.STATE][p_name] = self.exact_sol.pressure(sd, 0)
        data[pp.STATE][pp.ITERATE][p_name] = self.exact_sol.pressure(sd, 0)

        # Set initial displacement
        data[pp.STATE][u_name] = self.exact_sol.displacement(sd, 0)
        data[pp.STATE][pp.ITERATE][u_name] = self.exact_sol.displacement(sd, 0)

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False


class MandelSetup(  # type: ignore[misc]
    MandelGeometry,
    MandelPoromechanicsBoundaryConditions,
    MandelSolutionStrategy,
    MandelUtils,
    MandelDataSaving,
    BiotPoromechanics,
):
    """Mixer class for Mandel's consolidation problem.

    Model parameters of special relevance for this class:

        - vertical_load (pp.number): Applied load in [N * m^-1]. Default is 6e8.
        - domain_size (tuple(pp.number, pp.number)): Length and height of the domain
          in [m]. Default is (100, 10).
        - number_of_roots (int): Number of roots used to solve the exact solution.
          Default is 200.
        - material_constants (dict): Dictionary containing the solid and fluid
          constants. For suggested parameters, see ``mandel_solid_constants`` and
          ``mandel_fluid_constants`` at the top of file. Unitary/zero values are
          assigned by default. See below for the relevant material constants accessed
          by this verification.
        - time_manager (pp.TimeManager): Time manager object.
        - plot_results (bool): Whether to plot the results in non-dimensional form.
        - mesh_arguments (dict(str, number)): Mesh arguments in [m]. Default is
          2.0 * length_scaling.
        - units (pp.Units): Object containing scaling of base magnitudes. No scaling
          applied by default.

    Accessed material constants:

        - solid:
            - biot_coefficient  (Required value = 1)
            - lame_lambda
            - permeability
            - shear_modulus
            - specific_storage

        - fluid:
            - density
            - viscosity

    """
