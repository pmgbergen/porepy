"""
This module contains a code verification implementation for a manufactured solution for
the two-dimensional, compressible, single phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

The exact solution was obtained by extending the solution from the incompressible
case [1].

In particular, we have added a pressure-dependent density which obeys the following
constitutive relationship:

.. math:

    \\rho(p) = \\rho_0 \\exp{c_f (p - p_0)},

where :math:`\\rho` and :math:`p` are the density and pressure, :math:`\\rho_0` and
:math:`p_0` are the density and pressure at reference states, and :math:`c_f` is the
fluid compressibility.

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
      (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sym

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.viz.data_saving_model_mixin import VerificationDataSaving
from tests.functional.setups.manu_flow_incomp_frac_2d import (
    ManuIncompSaveData,
    ManuIncompUtils,
    SingleEmbeddedVerticalLineFracture,
)

# PorePy typings
number = pp.number
grid = pp.GridLike

# Material constants for the verification setup. Constants with (**) cannot be
# changed since the manufactured solution implicitly assumes such values.
manu_comp_fluid: dict[str, number] = {
    "viscosity": 1.0,  # (**)
    "compressibility": 0.2,
    "density": 1.0,  # reference value
    "pressure": 0.0,  # reference value
}

manu_comp_solid: dict[str, number] = {
    "normal_permeability": 0.5,  # (**) counteracts division by a/2 in interface law
    "permeability": 1.0,  # (**)
    "residual_aperture": 1.0,  # (**)
    "porosity": 0.1,  # reference value
}


# -----> Data-saving
@dataclass
class ManuCompSaveData(ManuIncompSaveData):
    """Data class to store relevant data from the verification setup."""

    time: number
    """Current simulation time."""


class ManuCompDataSaving(VerificationDataSaving):
    """Mixin class to store relevant data."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`porepy.models.constitutive_laws.DarcysLaw`.

    """

    exact_sol: ManuCompExactSolution2d
    """Exact solution object."""

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Darcy flux variables on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    def collect_data(self) -> ManuCompSaveData:
        """Collect data from the verification setup.

        Returns:
            ManuCompSaveData object containing the results of the verification for the
            current time.

        """

        sd_matrix: pp.Grid = self.mdg.subdomains()[0]
        sd_frac: pp.Grid = self.mdg.subdomains()[1]
        intf: pp.MortarGrid = self.mdg.interfaces()[0]
        exact_sol: ManuCompExactSolution2d = self.exact_sol
        t: number = self.time_manager.time

        # Collect data
        exact_matrix_pressure = exact_sol.matrix_pressure(sd_matrix, t)
        matrix_pressure_ad = self.pressure([sd_matrix])
        approx_matrix_pressure = matrix_pressure_ad.value(self.equation_system)
        error_matrix_pressure = ConvergenceAnalysis.l2_error(
            grid=sd_matrix,
            true_array=exact_matrix_pressure,
            approx_array=approx_matrix_pressure,
            is_scalar=True,
            is_cc=True,
            relative=True,
        )

        exact_matrix_flux = exact_sol.matrix_flux(sd_matrix, t)
        matrix_flux_ad = self.darcy_flux([sd_matrix])
        approx_matrix_flux = matrix_flux_ad.value(self.equation_system)
        error_matrix_flux = ConvergenceAnalysis.l2_error(
            grid=sd_matrix,
            true_array=exact_matrix_flux,
            approx_array=approx_matrix_flux,
            is_scalar=True,
            is_cc=False,
            relative=True,
        )

        exact_frac_pressure = exact_sol.fracture_pressure(sd_frac, t)
        frac_pressure_ad = self.pressure([sd_frac])
        approx_frac_pressure = frac_pressure_ad.value(self.equation_system)
        error_frac_pressure = ConvergenceAnalysis.l2_error(
            grid=sd_frac,
            true_array=exact_frac_pressure,
            approx_array=approx_frac_pressure,
            is_scalar=True,
            is_cc=True,
            relative=True,
        )

        exact_frac_flux = exact_sol.fracture_flux(sd_frac, t)
        frac_flux_ad = self.darcy_flux([sd_frac])
        approx_frac_flux = frac_flux_ad.value(self.equation_system)
        error_frac_flux = ConvergenceAnalysis.l2_error(
            grid=sd_frac,
            true_array=exact_frac_flux,
            approx_array=approx_frac_flux,
            is_scalar=True,
            is_cc=False,
            relative=True,
        )

        exact_intf_flux = exact_sol.interface_flux(intf, t)
        int_flux_ad = self.interface_darcy_flux([intf])
        approx_intf_flux = int_flux_ad.value(self.equation_system)
        error_intf_flux = ConvergenceAnalysis.l2_error(
            grid=intf,
            true_array=exact_intf_flux,
            approx_array=approx_intf_flux,
            is_scalar=True,
            is_cc=True,
            relative=True,
        )

        # Store collected data in data class
        collected_data = ManuCompSaveData(
            approx_frac_flux=approx_frac_flux,
            approx_frac_pressure=approx_frac_pressure,
            approx_intf_flux=approx_intf_flux,
            approx_matrix_flux=approx_matrix_flux,
            approx_matrix_pressure=approx_matrix_pressure,
            error_frac_flux=error_frac_flux,
            error_frac_pressure=error_frac_pressure,
            error_intf_flux=error_intf_flux,
            error_matrix_flux=error_matrix_flux,
            error_matrix_pressure=error_matrix_pressure,
            exact_frac_flux=exact_frac_flux,
            exact_frac_pressure=exact_frac_pressure,
            exact_intf_flux=exact_intf_flux,
            exact_matrix_flux=exact_matrix_flux,
            exact_matrix_pressure=exact_matrix_pressure,
            time=t,
        )

        return collected_data


# -----> Exact solution
class ManuCompExactSolution2d:
    """Class containing the exact manufactured solution for the verification setup."""

    def __init__(self, setup):
        """Constructor of the class."""

        # Retrieve material constant from the setup
        rho_0 = setup.fluid.reference_component.density  # [kg * m^-3]  Reference fluid density
        p_0 = setup.fluid.reference_component.pressure  # [Pa] Reference fluid pressure
        c_f = setup.fluid.reference_component.compressibility  # [Pa^-1]  Fluid compressibility
        phi_0 = setup.solid.porosity  # [-] Reference porosity

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Smoothness coefficient
        n = 1.5

        # Distance and bubble functions
        distance_fun = [
            ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5,  # bottom
            ((x - 0.5) ** 2) ** 0.5,  # middle
            ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5,  # top
        ]
        bubble_fun = (y - 0.25) ** 2 * (y - 0.75) ** 2

        # Exact pressure in the matrix
        p_matrix = [
            t * (distance_fun[0] ** (1 + n)),
            t * (distance_fun[1] ** (1 + n) + bubble_fun * distance_fun[1]),
            t * (distance_fun[2] ** (1 + n)),
        ]

        # Exact Darcy flux in the matrix
        q_matrix = [[-sym.diff(p, x), -sym.diff(p, y)] for p in p_matrix]

        # Exact density in the matrix
        rho_matrix = [rho_0 * sym.exp(c_f * (p - p_0)) for p in p_matrix]

        # Exact mass flux in the matrix
        mf_matrix = [[rho * q[0], rho * q[1]] for (rho, q) in zip(rho_matrix, q_matrix)]

        # Exact divergence of the mass flux in the matrix
        div_mf_matrix = [sym.diff(mf[0], x) + sym.diff(mf[1], y) for mf in mf_matrix]

        # Exact accumulation term in the matrix
        accum_matrix = [sym.diff(phi_0 * rho, t) for rho in rho_matrix]

        # Exact source term in the matrix
        f_matrix = [
            accum + div_mf for (accum, div_mf) in zip(accum_matrix, div_mf_matrix)
        ]

        # Exact flux on the interface (mortar fluxes)
        q_intf = t * bubble_fun

        # Exact density on the interface
        rho_intf = 1.0  # density evaluates to 1 at x = 0.5, 0.25 <= y <= 0.75

        # Exact mass flux on the interface
        mf_intf = rho_intf * q_intf

        # Exact pressure in the fracture
        p_frac = -t * bubble_fun

        # Exact Darcy flux in the fracture
        q_frac = -sym.diff(p_frac, y)

        # Exact density in the fracture
        rho_frac = rho_0 * sym.exp(c_f * (p_frac - p_0))

        # Exact mass flux in the fracture
        mf_frac = rho_frac * q_frac

        # Exact accumulation in the fracture
        accum_frac = sym.diff(phi_0 * rho_frac, t)

        # Exact divergence of the mass flux in the fracture
        div_mf_frac = sym.diff(mf_frac, y)

        # Exact source in the fracture
        f_frac = accum_frac + div_mf_frac - 2 * mf_intf

        # Public attributes
        self.p_matrix = p_matrix
        self.q_matrix = q_matrix
        self.rho_matrix = rho_matrix
        self.mf_matrix = mf_matrix
        self.f_matrix = f_matrix

        self.p_frac = p_frac
        self.q_frac = q_frac
        self.rho_frac = rho_frac
        self.mf_frac = mf_frac
        self.f_frac = f_frac

        self.q_intf = q_intf
        self.rho_intf = rho_intf
        self.mf_intf = mf_intf

        # Private attributes
        self._bubble = bubble_fun

    def matrix_pressure(self, sd_matrix: pp.Grid, time: number) -> np.ndarray:
        """Evaluate exact matrix pressure [Pa] at the cell centers.

        Parameters:
            sd_matrix: Matrix grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact pressures at
            the cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, t), p, "numpy") for p in self.p_matrix]

        # Cell-centered pressures
        p_cc = np.zeros(sd_matrix.num_cells)
        for p, idx in zip(p_fun, cell_idx):
            p_cc += p(cc[0], cc[1], time) * idx

        return p_cc

    def matrix_flux(self, sd_matrix: pp.Grid, time: number) -> np.ndarray:
        """Evaluate exact matrix Darcy flux [m^3 * s^-1] at the face centers .

        Parameters:
            sd_matrix: Matrix grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` containing the exact Darcy
            fluxes at the face centers at the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd_matrix.face_normals``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd_matrix.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify bubble function
        bubble_fun = sym.lambdify((y, t), self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [
                sym.lambdify((x, y, t), q[0], "numpy"),
                sym.lambdify((x, y, t), q[1], "numpy"),
            ]
            for q in self.q_matrix
        ]

        # Computation of the fluxes in the middle region results in NaN on faces that
        # are outside the middle region. We therefore need to first evaluate the middle
        # region and then the other regions, so that NaN faces outside the middle
        # region can be overwritten accordingly.
        fn = sd_matrix.face_normals
        q_fc = np.zeros(sd_matrix.num_faces)

        q_fun_sorted = q_fun.copy()
        q_fun_sorted.pop(1)
        q_fun_sorted.insert(0, q_fun[1])

        face_idx_sorted = face_idx.copy()
        face_idx_sorted.pop(1)
        face_idx_sorted.insert(0, face_idx[1])

        # Perform evaluations using the sorted list of exact Darcy velocities
        for q, idx in zip(q_fun_sorted, face_idx_sorted):
            q_fc[idx] = (
                q[0](fc[0][idx], fc[1][idx], time) * fn[0][idx]
                + q[1](fc[0][idx], fc[1][idx], time) * fn[1][idx]
            )

        # We need to correct the values of the exact Darcy fluxes at the internal
        # boundaries since they evaluate to NaN due to a division by zero (this
        # happens because the distance function evaluates to zero on internal
        # boundaries).

        # For the correction, we exploit the fact that (rho_matrix * q_matrix) \dot n =
        # (rho_intf * q_intf) holds in a continuous sense. Furthermore, for our
        # problem, rho_matrix = rho_intf = 1.0 at x = 0.5 and 0.25 <= y <= 0.75 . Thus,
        # the previous equality can be simplified to q_matrix \dot n = q_intf on the
        # internal boundaries.

        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_matrix.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces], time)
            * sd_matrix.face_areas[frac_faces]
            * sd_matrix.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def matrix_source(self, sd_matrix: pp.Grid, time: number) -> np.ndarray:
        """Compute exact integrated matrix source.

        Parameters:
            sd_matrix: Matrix grid.
            time: float

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact integrated
            sources at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y, t), f, "numpy") for f in self.f_matrix]

        # Integrated cell-centered sources
        vol = sd_matrix.cell_volumes
        f_cc = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            f_cc += f(cc[0], cc[1], time) * vol * idx

        return f_cc

    def fracture_pressure(self, sd_frac: pp.Grid, time: number) -> np.ndarray:
        """Evaluate exact fracture pressure at the cell centers.

        Parameters:
            sd_frac: Fracture grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact pressures at
            the cell centers at the given ``time``.

        """
        # Symbolic variable
        y, t = sym.symbols("y t")

        # Cell centers
        cc = sd_frac.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify((y, t), self.p_frac, "numpy")

        # Evaluate at the cell centers
        p_cc = p_fun(cc[1], time)

        return p_cc

    def fracture_flux(self, sd_frac: pp.Grid, time: number) -> np.ndarray:
        """Evaluate exact fracture Darcy flux at the face centers.

        Parameters:
            sd_frac: Fracture grid.
            time: float

        Returns:
            Array of ``shape=(sd_frac.num_faces, )`` containing the exact Darcy
            fluxes at the face centers at the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd_face.face_normals``.

        """
        # Symbolic variable
        y, t = sym.symbols("y t")

        # Face centers and face normals
        fc = sd_frac.face_centers
        fn = sd_frac.face_normals

        # Lambdify expression
        q_fun = sym.lambdify((y, t), self.q_frac, "numpy")

        # Evaluate at the face centers and scale with face normals
        q_fc = q_fun(fc[1], time) * fn[1]

        return q_fc

    def fracture_source(self, sd_frac: pp.Grid, time: number) -> np.ndarray:
        """Compute exact integrated fracture source.

        Parameters:
            sd_frac: Fracture grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact integrated
            sources for a the given ``time``.

        """
        # Symbolic variable
        y, t = sym.symbols("y t")

        # Cell centers and volumes
        cc = sd_frac.cell_centers
        vol = sd_frac.cell_volumes

        # Lambdify expression
        f_fun = sym.lambdify((y, t), self.f_frac, "numpy")

        # Evaluate and integrate
        f_cc = f_fun(cc[1], time) * vol

        return f_cc

    def interface_flux(self, intf: pp.MortarGrid, time: number) -> np.ndarray:
        """Compute exact mortar fluxes at the interface.

        Parameters:
            intf: Mortar grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(intf.num_cells, )`` containing the exact mortar fluxes
            at the given ``time``.

        Note:
            The returned mortar fluxes are already scaled with ``intf.cell_volumes``.

        """
        # Symbolic variable
        y, t = sym.symbols("y t")

        # Cell centers and volumes
        cc = intf.cell_centers
        vol = intf.cell_volumes

        # Lambdify expression
        lmbda_fun = sym.lambdify((y, t), self.q_intf, "numpy")

        # Evaluate and integrate
        lmbda_cc = lmbda_fun(cc[1], time) * vol

        return lmbda_cc

    def matrix_boundary_pressure(
        self, bg_matrix: pp.BoundaryGrid, time: number
    ) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            bg_matrix: Boundary grid of the matrix.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` with the exact pressure values
            on the exterior boundary faces at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = bg_matrix.cell_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, t), p, "numpy") for p in self.p_matrix]

        # Boundary pressures
        p_bf = np.zeros(bg_matrix.num_cells)
        for p, idx in zip(p_fun, face_idx):
            p_bf += p(fc[0], fc[1], time) * idx

        return p_bf


# -----> Boundary conditions
class ManuCompBoundaryConditions(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow
):
    """Set boundary conditions for the simulation model."""

    exact_sol: ManuCompExactSolution2d
    """Exact solution object."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Set boundary condition types for the elliptic discretization."""
        if sd.dim == self.mdg.dim_max():  # Dirichlet for the matrix
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        else:  # Neumann for the fracture tips
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Set boundary condition types for the upwind discretization."""
        if sd.dim == self.mdg.dim_max():  # Dirichlet for the matrix
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        else:  # Neumann for the fracture tips
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Analytical boundary condition values for Darcy flux.

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        # Retrieve current time
        t = self.time_manager.time

        sd_matrix = self.mdg.subdomains()[0]
        sd_frac = self.mdg.subdomains()[1]

        # Boundary conditions for the elliptic discretization
        if boundary_grid.parent == sd_matrix:
            return self.exact_sol.matrix_boundary_pressure(boundary_grid, t)
        elif boundary_grid.parent == sd_frac:
            return np.zeros(boundary_grid.num_cells)
        else:
            raise ValueError(boundary_grid)


# -----> Balance equations
class ManuCompBalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Modify balance equation to account for time-dependent external sources."""

        # Internal sources are inherit from parent class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # External sources are retrieved from SOLUTIONS and wrapped as an
        # AdArray.
        external_sources = pp.ad.TimeDependentDenseArray(
            name="external_sources",
            domains=self.mdg.subdomains(),
        ).previous_timestep()

        # Add-up contribution
        fluid_source = internal_sources + external_sources
        fluid_source.set_name("Time-dependent fluid source")

        return fluid_source


# -----> Solution strategy
class ManuCompSolutionStrategy2d(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`porepy.models.constitutive_laws.DarcysLaw`.

    """

    exact_sol: ManuCompExactSolution2d
    """Exact solution object."""

    plot_results: Callable
    """Method to plot results of the verification setup. Usually provided by the
    mixin class :class:`SetupUtilities`.

    """

    results: list[ManuCompSaveData]
    """List of SaveData objects."""

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    def __init__(self, params: dict):
        """Constructor of the class."""
        super().__init__(params)

        self.exact_sol: ManuCompExactSolution2d
        """Exact solution object."""

        self.results: list[ManuCompSaveData] = []
        """Object that stores exact and approximated solutions and L2 errors."""

        self.subdomain_darcy_flux_variable: str = "darcy_flux"
        """Keyword to access the subdomain Darcy fluxes."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks
        assert self.fluid.reference_component.viscosity == 1
        assert self.solid.permeability == 1
        assert self.solid.residual_aperture == 1
        assert self.solid.normal_permeability == 0.5

        # Instantiate exact solution object
        self.exact_sol = ManuCompExactSolution2d(self)

    def update_time_dependent_ad_arrays(self) -> None:
        """Update values of external sources and boundary conditions."""
        super().update_time_dependent_ad_arrays()

        # Retrieve subdomains and data dictionaries
        sd_matrix = self.mdg.subdomains()[0]
        data_matrix = self.mdg.subdomain_data(sd_matrix)
        sd_frac = self.mdg.subdomains()[1]
        data_frac = self.mdg.subdomain_data(sd_frac)

        # Retrieve current time
        t = self.time_manager.time

        # Sources
        matrix_source = self.exact_sol.matrix_source(sd_matrix, t)

        pp.set_solution_values(
            name="external_sources",
            values=matrix_source,
            data=data_matrix,
            time_step_index=0,
        )

        frac_source = self.exact_sol.fracture_source(sd_frac, t)

        pp.set_solution_values(
            name="external_sources",
            values=frac_source,
            data=data_frac,
            time_step_index=0,
        )

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()


# -----> Mixer
class ManuCompFlowSetup2d(  # type: ignore[misc]
    SingleEmbeddedVerticalLineFracture,
    ManuCompBalanceEquation,
    ManuCompBoundaryConditions,
    ManuCompSolutionStrategy2d,
    ManuIncompUtils,
    ManuCompDataSaving,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """
    Mixer class for the 2d compressible flow setup with a single fracture.
    """
