"""
This module contains a code verification implementation for a manufactured solution for
the two-dimensional, compressible, single phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

The exact solution was obtained by extending the solution from the incompressible
case [1].

In particular, we have added a pressure-dependent density which obeys the following
consitutive relationship:

.. math:

    \rho(p) = \rho_0 \exp{c_f (p - p_0)},

where :math:`\rho` and :math:`p` are the density and pressure, :math:`\rho_0` and
:math:`p_0` are the density and pressure at reference states, and :math:`c_f` is the
fluid compressibility.

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
      (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""
from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sym

import porepy as pp
from porepy.applications.verification_setups.manu_flow_incomp_frac import (
    SaveData,
    SetupUtilities,
    SingleEmbeddedVerticalFracture,
)
from porepy.applications.verification_setups.verification_utils import VerificationUtils

# PorePy typings
number = pp.number
grid = pp.GridLike

# Material constants for the verification setup. Constants with (**) cannot be
# changed since the manufactured solution implicitly assume such values.
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


class ModifiedDataSavingMixin(pp.DataSavingMixin):
    """Mixin class to store relevant data."""

    exact_sol: ManuCompExactSolution
    """Exact solution object."""

    interface_darcy_flux_variable: str
    """Key to access the interface variable."""

    pressure_variable: str
    """Key to access the pressure variable."""

    relative_l2_error: Callable
    """Method that computes the discrete relative L2-error between an exact solution
    and an approximate solution on a given grid. The method is provided by the mixin
    class :class:`porepy.applications.verification_setups.VerificationUtils`.

    """

    results: list[SaveData]
    """List of :class:`SaveData` objects containing the results of the verification."""

    subdomain_darcy_flux_variable: str
    """Keyword to access the subdomain Darcy fluxes."""

    def save_data_time_step(self) -> None:
        """Save data to the `results` list.

        Note:
            Data will be appended to the ``results`` list only if the current time
            matches a time from ``self.time_manager.schedule[1:]``.

        """
        if any(np.isclose(self.time_manager.time, self.time_manager.schedule[1:])):
            collected_data = self._collect_data()
            self.results.append(collected_data)

    def _collect_data(self) -> SaveData:
        """Collect data from the verification setup.

        Returns:
            SaveData object containing the results of the verification for the
            current time.

        """

        sd_rock: pp.Grid = self.mdg.subdomains()[0]
        data_rock: dict = self.mdg.subdomain_data(sd_rock)
        sd_frac: pp.Grid = self.mdg.subdomains()[1]
        data_frac: dict = self.mdg.subdomain_data(sd_frac)
        intf: pp.MortarGrid = self.mdg.interfaces()[0]
        data_intf: dict = self.mdg.interface_data(intf)
        p_name: str = self.pressure_variable
        q_name: str = self.subdomain_darcy_flux_variable
        lmbda_name: str = self.interface_darcy_flux_variable
        exact_sol: ManuCompExactSolution = self.exact_sol
        t: number = self.time_manager.time

        # Instantiate data class
        out = SaveData()

        # Rock pressure
        out.exact_rock_pressure = exact_sol.rock_pressure(sd_rock, t)
        out.approx_rock_pressure = data_rock[pp.STATE][pp.ITERATE][p_name].copy()
        out.error_rock_pressure = self.relative_l2_error(
            grid=sd_rock,
            true_array=out.exact_rock_pressure,
            approx_array=out.approx_rock_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Fracture pressure
        out.exact_frac_pressure = exact_sol.fracture_pressure(sd_frac, t)
        out.approx_frac_pressure = data_frac[pp.STATE][pp.ITERATE][p_name].copy()
        out.error_frac_pressure = self.relative_l2_error(
            grid=sd_frac,
            true_array=out.exact_frac_pressure,
            approx_array=out.approx_frac_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Rock flux
        out.exact_rock_flux = exact_sol.rock_flux(sd_rock, t)
        out.approx_rock_flux = data_rock[pp.STATE][pp.ITERATE][q_name].copy()
        out.error_rock_flux = self.relative_l2_error(
            grid=sd_rock,
            true_array=out.exact_rock_flux,
            approx_array=out.approx_rock_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Fracture flux
        out.exact_frac_flux = exact_sol.fracture_flux(sd_frac, t)
        out.approx_frac_flux = data_frac[pp.STATE][pp.ITERATE][q_name].copy()
        out.error_frac_flux = self.relative_l2_error(
            grid=sd_frac,
            true_array=out.exact_frac_flux,
            approx_array=out.approx_frac_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Interface flux
        out.exact_intf_flux = exact_sol.interface_flux(intf, t)
        out.approx_intf_flux = data_intf[pp.STATE][pp.ITERATE][lmbda_name]
        out.error_intf_flux = self.relative_l2_error(
            grid=intf,
            true_array=out.exact_intf_flux,
            approx_array=out.approx_intf_flux,
            is_scalar=True,
            is_cc=True,
        )

        return out


class ManuCompExactSolution:
    """Class containing the exact solution for the verification setup."""

    def __init__(self, setup):
        """Constructor of the class."""

        # Retrieve material constant from the setup
        rho_0 = setup.fluid.density()  # [kg * m^-3]  Reference fluid density
        p_0 = setup.fluid.pressure()  # [Pa] Reference fluid pressure
        c_f = setup.fluid.compressibility()  # [Pa^-1]  Fluid compressibility
        phi_0 = setup.solid.porosity()  # [-] Reference porosity

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

        # Exact pressure in the rock
        p_rock = [
            t * (distance_fun[0] ** (1 + n)),
            t * (distance_fun[1] ** (1 + n) + bubble_fun * distance_fun[1]),
            t * (distance_fun[2] ** (1 + n)),
        ]

        # Exact Darcy flux in the rock
        q_rock = [[-sym.diff(p, x), -sym.diff(p, y)] for p in p_rock]

        # Exact density in the rock
        rho_rock = [rho_0 * sym.exp(c_f * (p - p_0)) for p in p_rock]

        # Exact mass flux in the rock
        mf_rock = [[rho * q[0], rho * q[1]] for (rho, q) in zip(rho_rock, q_rock)]

        # Exact divergence of the mass flux in the rock
        div_mf_rock = [sym.diff(mf[0], x) + sym.diff(mf[1], y) for mf in mf_rock]

        # Exact accumulation term in the rock
        accum_rock = [sym.diff(phi_0 * rho, t) for rho in rho_rock]

        # Exact source term in the rock
        f_rock = [accum + div_mf for (accum, div_mf) in zip(accum_rock, div_mf_rock)]

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
        self.p_rock = p_rock
        self.q_rock = q_rock
        self.rho_rock = rho_rock
        self.mf_rock = mf_rock
        self.f_rock = f_rock

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

    def rock_pressure(self, sd_rock: pp.Grid, time: number) -> np.ndarray:
        """Evaluate exact rock pressure [Pa] at the cell centers.

        Parameters:
            sd_rock: Rock grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact pressures at
            the cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of cell indices
        cc = sd_rock.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, t), p, "numpy") for p in self.p_rock]

        # Cell-centered pressures
        p_cc = np.zeros(sd_rock.num_cells)
        for (p, idx) in zip(p_fun, cell_idx):
            p_cc += p(cc[0], cc[1], time) * idx

        return p_cc

    def rock_flux(self, sd_rock: pp.Grid, time: number) -> np.ndarray:
        """Evaluate exact rock Darcy flux [m^3 * s^-1] at the face centers .

        Parameters:
            sd_rock: Rock grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` containing the exact Darcy
            fluxes at the face centers at the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd_rock.face_normals``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd_rock.face_centers
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
            for q in self.q_rock
        ]

        # Face-centered Darcy fluxes
        fn = sd_rock.face_normals
        q_fc = np.zeros(sd_rock.num_faces)
        for (q, idx) in zip(q_fun, face_idx):
            q_fc += (
                q[0](fc[0], fc[1], time) * fn[0] + q[1](fc[0], fc[1], time) * fn[1]
            ) * idx

        # We need to correct the values of the exact Darcy fluxes at the internal
        # boundaries since they evaluate to NaN due to a division by zero (this
        # happens because the distance function evaluates to zero on internal
        # boundaries).

        # For the correction, we exploit the fact that (rho_rock * q_rock) \dot n =
        # (rho_intf * q_intf) holds in a continuous sense. Furthermore, for our
        # problem, rho_rock = rho_intf = 1.0 at x = 0.5 and 0.25 <= y <= 0.75 . Thus,
        # the previous equality can be simplified to q_rock \dot n = q_intf on the
        # internal boundaries.

        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_rock.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces], time)
            * sd_rock.face_areas[frac_faces]
            * sd_rock.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def rock_source(self, sd_rock: pp.Grid, time: number) -> np.ndarray:
        """Compute exact integrated rock source.

        Parameters:
            sd_rock: Rock grid.
            time: float

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact integrated
            sources at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of cell indices
        cc = sd_rock.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y, t), f, "numpy") for f in self.f_rock]

        # Integrated cell-centered sources
        vol = sd_rock.cell_volumes
        f_cc = np.zeros(sd_rock.num_cells)
        for (f, idx) in zip(f_fun, cell_idx):
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

    def rock_boundary_pressure(self, sd_rock: pp.Grid, time: number) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            sd_rock: Rock grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` with the exact pressure values
            on the exterior boundary faces at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd_rock.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Boundary faces
        bc_faces = sd_rock.get_boundary_faces()

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, t), p, "numpy") for p in self.p_rock]

        # Boundary pressures
        p_bf = np.zeros(sd_rock.num_faces)
        for (p, idx) in zip(p_fun, face_idx):
            p_bf[bc_faces] += p(fc[0], fc[1], time)[bc_faces] * idx[bc_faces]

        return p_bf

    def rock_boundary_density(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Exact density at the boundary faces.

        Parameters:
            sd_rock: Rock grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` with the exact density values
            on the exterior boundary faces for the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd_rock.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Boundary faces
        bc_faces = sd_rock.get_boundary_faces()

        # Lambdify expression
        rho_fun = [sym.lambdify((x, y, t), rho, "numpy") for rho in self.rho_rock]

        # Boundary pressures
        rho_bf = np.zeros(sd_rock.num_faces)
        for (rho, idx) in zip(rho_fun, face_idx):
            rho_bf[bc_faces] += rho(fc[0], fc[1], time)[bc_faces] * idx[bc_faces]

        return rho_bf


class ManuCompBoundaryConditions:
    """Set boundary conditions for the simulation model."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid object."""

    time_manager: pp.TimeManager
    """Properly initialized time manager object."""

    domain_boundary_sides: Callable[[pp.Grid], pp.bounding_box.DomainSides]
    """Utility function to access the domain boundary sides."""

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Set boundary condition types for the elliptic discretization."""
        if sd.dim == 2:  # Dirichlet for the rock
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        else:  # Neumann for the fracture tips
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")

    def bc_type_mobrho(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Set boundary condition types for the upwind discretization."""
        if sd.dim == 2:  # Dirichlet for the rock
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        else:  # Neumann for the fracture tips
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Set boundary condition values for the elliptic discretization."""
        bc_values = pp.ad.TimeDependentArray(
            name="darcy_bc_values", subdomains=subdomains, previous_timestep=True
        )
        return bc_values

    def bc_values_mobrho(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Set boundary condition values for the upwind discretization."""
        bc_values = pp.ad.TimeDependentArray(
            name="mobrho_bc_values", subdomains=subdomains, previous_timestep=True
        )
        return bc_values


class ManuCompBalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Modify balance equation to account for time-dependent external sources."""

        # Internal sources are inherit from parent class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # External sources are retrieved from STATE and wrapped as an AdArray
        external_sources = pp.ad.TimeDependentArray(
            name="external_sources",
            subdomains=self.mdg.subdomains(),
            previous_timestep=True,
        )

        # Add-up contribution
        fluid_source = internal_sources + external_sources
        fluid_source.set_name("Time-dependent fluid source")

        return fluid_source


class ManuCompSolutionStrategy(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`porepy.models.constitutive_laws.DarcysLaw`.

    """

    exact_sol: ManuCompExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants
    """Object containing the fluid constants."""

    plot_results: Callable
    """Method to plot results of the verification setup. Usually provided by the
    mixin class :class:`SetupUtilities`.

    """

    results: list[SaveData]
    """List of SaveData objects."""

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    def __init__(self, params: dict):
        """Constructor of the class."""
        super().__init__(params)

        self.exact_sol: ManuCompExactSolution
        """Exact solution object."""

        self.results: list[SaveData] = []
        """Object that stores exact and approximated solutions and L2 errors."""

        self.subdomain_darcy_flux_variable: str = "darcy_flux"
        """Keyword to access the subdomain Darcy fluxes."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks
        assert self.fluid.viscosity() == 1
        assert self.solid.permeability() == 1
        assert self.solid.residual_aperture() == 1
        assert self.solid.normal_permeability() == 0.5

        # Instantiate exact solution object
        self.exact_sol = ManuCompExactSolution(self)

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources and boundary conditions."""

        # Retrieve subdomains and data dictionaries
        sd_rock = self.mdg.subdomains()[0]
        data_rock = self.mdg.subdomain_data(sd_rock)
        sd_frac = self.mdg.subdomains()[1]
        data_frac = self.mdg.subdomain_data(sd_frac)

        # Retrieve current time
        t = self.time_manager.time

        # Sources
        rock_source = self.exact_sol.rock_source(sd_rock, t)
        data_rock[pp.STATE]["external_sources"] = rock_source

        frac_source = self.exact_sol.fracture_source(sd_frac, t)
        data_frac[pp.STATE]["external_sources"] = frac_source

        # Boundary conditions for the elliptic discretization
        rock_pressure_boundary = self.exact_sol.rock_boundary_pressure(sd_rock, t)
        data_rock[pp.STATE]["darcy_bc_values"] = rock_pressure_boundary
        data_frac[pp.STATE]["darcy_bc_values"] = np.zeros(sd_frac.num_faces)

        # Boundary conditions for the upwind discretization
        rock_density_boundary = self.exact_sol.rock_boundary_density(sd_rock, t)
        viscosity = self.fluid.viscosity()
        rock_mobrho = rock_density_boundary / viscosity
        data_rock[pp.STATE]["mobrho_bc_values"] = rock_mobrho
        data_frac[pp.STATE]["mobrho_bc_values"] = np.zeros(sd_frac.num_faces)

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after convergence."""

        # Store subdomain Darcy fluxes in pp.STATE
        for sd, data in self.mdg.subdomains(return_data=True):
            darcy_flux_ad = self.darcy_flux([sd])
            vals = darcy_flux_ad.evaluate(self.equation_system).val
            data[pp.STATE][pp.ITERATE][self.subdomain_darcy_flux_variable] = vals

        # Inherit from base class
        super().after_nonlinear_convergence(solution, errors, iteration_counter)

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()


class ManufacturedCompressibleFlow2d(  # type: ignore[misc]
    ManuCompBalanceEquation,
    ManuCompBoundaryConditions,
    ManuCompSolutionStrategy,
    pp.fluid_mass_balance.SinglePhaseFlow,
    SetupUtilities,
    VerificationUtils,
    SingleEmbeddedVerticalFracture,
    ModifiedDataSavingMixin,
):
    """
    Mixer class for the compressible flow with a single fracture verification setup.

    Examples:

        .. code:: python

            from time import time
            from porepy.applications.verification_setups.manu_flow_comp_frac import (
                manu_comp_solid,
                manu_comp_fluid,
            )

            # Simulation parameters
            material_constants = {
                "solid": pp.SolidConstants(manu_comp_solid),
                "fluid": pp.FluidConstants(manu_comp_fluid),
            }

            time_manager = pp.TimeManager(
                schedule=[0, 0.2, 0.6, 1.0], dt_init=0.2, constant_dt=True
            )

            params = {
                "material_constants": material_constants,
                "plot_results": True,
                "time_manager": time_manager,
            }

            # Run verification setup
            tic = time()
            setup = ManufacturedCompressibleFlow2d(params)
            print("Simulation started...")
            pp.run_time_dependent_model(setup, params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds.")

    """
