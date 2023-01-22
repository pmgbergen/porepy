"""
This module contains a code verification implementation using a manufactured solution
for the two-dimensional, incompressible, single phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

Details regarding the manufactured solution can be found in Appendix D.1 from [1].

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
      (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import porepy as pp
from porepy.applications.verification_setups.verification_utils import VerificationUtils

# PorePy typings
number = pp.number
grid = pp.GridLike

# Material constants for the verification setup. Constants with (**) cannot be
# changed since the manufactured solution implicitly assume such values.
manu_incomp_fluid: dict[str, number] = {
    "compressibility": 0,  # (**)
    "density": 1.0,  # (**)
    "viscosity": 1.0,  # (**)
}

manu_incomp_solid: dict[str, number] = {
    "residual_aperture": 1.0,  # (**)
    "permeability": 1.0,  # (**)
    "normal_permeability": 0.5,  # (**) counteracts division by a/2 in interface law
}


@dataclass
class SaveData:
    """Data class to save relevant results from the verification setup."""

    approx_frac_flux: np.ndarray = np.zeros(1)
    """Numerical flux in the fracture."""

    approx_frac_pressure: np.ndarray = np.zeros(1)
    """Numerical pressure in the fracture."""

    approx_intf_flux: np.ndarray = np.zeros(1)
    """Numerical flux on the interfaces."""

    approx_rock_flux: np.ndarray = np.zeros(1)
    """Numerical flux in the rock."""

    approx_rock_pressure: np.ndarray = np.zeros(1)
    """Numerical pressure in the rock."""

    error_frac_flux: number = 0
    """L2-discrete relative error for the flux in the fracture."""

    error_frac_pressure: number = 0
    """L2-discrete relative error for the pressure in the fracture."""

    error_intf_flux: number = 0
    """L2-discrete relative error for the flux on the interfaces."""

    error_rock_flux: number = 0
    """L2-discrete relative error for the flux in the rock."""

    error_rock_pressure: number = 0
    """L2-discrete relative error for the pressure in the rock."""

    exact_frac_flux: np.ndarray = np.zeros(1)
    """Exact flux in the fracture."""

    exact_frac_pressure: np.ndarray = np.zeros(1)
    """Exact pressure in the fracture."""

    exact_intf_flux: np.ndarray = np.zeros(1)
    """Exact flux on the interfaces."""

    exact_rock_flux: np.ndarray = np.zeros(1)
    """Exact flux in the rock."""

    exact_rock_pressure: np.ndarray = np.zeros(1)
    """Exact pressure in the rock."""


class ManuIncompExactSolution:
    """Parent class for the exact solution."""

    def __init__(self):
        """Constructor of the class."""

        # Symbolic variables
        x, y = sym.symbols("x y")

        # Smoothness exponent
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
            distance_fun[0] ** (1 + n),
            distance_fun[1] ** (1 + n) + bubble_fun * distance_fun[1],
            distance_fun[2] ** (1 + n),
        ]

        # Exact Darcy flux in the rock
        q_rock = [[-sym.diff(p, x), -sym.diff(p, y)] for p in p_rock]

        # Exact source in the rock
        f_rock = [sym.diff(q[0], x) + sym.diff(q[1], y) for q in q_rock]

        # Exact interface flux
        q_intf = bubble_fun

        # Exact pressure in the fracture
        p_frac = -bubble_fun

        # Exact Darcy flux in the fracture
        q_frac = -sym.diff(p_frac, y)

        # Exact source in the fracture
        f_frac = sym.diff(q_frac, y) - 2 * q_intf

        # Public attributes
        self.p_rock = p_rock
        self.q_rock = q_rock
        self.f_rock = f_rock
        self.p_frac = p_frac
        self.q_frac = q_frac
        self.f_frac = f_frac
        self.q_intf = q_intf

        # Private attributes
        self._bubble = bubble_fun

    def rock_pressure(self, sd_rock: pp.Grid) -> np.ndarray:
        """Evaluate exact rock pressure at the cell centers.

        Parameters:
            sd_rock: Rock grid.

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_rock.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        p_fun = [sym.lambdify((x, y), p, "numpy") for p in self.p_rock]

        # Cell-centered pressures
        p_cc = np.zeros(sd_rock.num_cells)
        for (p, idx) in zip(p_fun, cell_idx):
            p_cc += p(cc[0], cc[1]) * idx

        return p_cc

    def rock_flux(self, sd_rock: pp.Grid) -> np.ndarray:
        """Evaluate exact rock Darcy flux at the face centers.

        Parameters:
            sd_rock: Rock grid.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_rock.face_normals``.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of face indices
        fc = sd_rock.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify bubble
        bubble_fun = sym.lambdify(y, self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [sym.lambdify((x, y), q[0], "numpy"), sym.lambdify((x, y), q[1], "numpy")]
            for q in self.q_rock
        ]

        # Face-centered Darcy fluxes
        fn = sd_rock.face_normals
        q_fc = np.zeros(sd_rock.num_faces)
        for (q, idx) in zip(q_fun, face_idx):
            q_fc += (q[0](fc[0], fc[1]) * fn[0] + q[1](fc[0], fc[1]) * fn[1]) * idx

        # We need to correct the values of the exact Darcy fluxes at the internal
        # boundaries since they evaluate to NaN due to a division by zero.
        # What we do is to exploit the fact that trace(q) = \lambda holds in a
        # continuous sense, and use that expression on internal boundaries instead.
        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_rock.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces])
            * sd_rock.face_areas[frac_faces]
            * sd_rock.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def rock_source(self, sd_rock: pp.Grid) -> np.ndarray:
        """Compute exact integrated rock source.

        Parameters:
            sd_rock: Rock grid.

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact integrated
            sources.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_rock.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y), f, "numpy") for f in self.f_rock]

        # Integrated cell-centered sources
        vol = sd_rock.cell_volumes
        f_cc = np.zeros(sd_rock.num_cells)
        for (f, idx) in zip(f_fun, cell_idx):
            f_cc += f(cc[0], cc[1]) * vol * idx

        return f_cc

    def fracture_pressure(self, sd_frac: pp.Grid) -> np.ndarray:
        """Evaluate exact fracture pressure at the cell centers.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Cell centers
        cc = sd_frac.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify(y, self.p_frac, "numpy")

        # Evaluate at the cell centers
        p_cc = p_fun(cc[1])

        return p_cc

    def fracture_flux(self, sd_frac: pp.Grid) -> np.ndarray:
        """Evaluate exact fracture Darcy flux at the face centers.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_face.face_normals``.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Face centers and face normals
        fc = sd_frac.face_centers
        fn = sd_frac.face_normals

        # Lambdify expression
        q_fun = sym.lambdify(y, self.q_frac, "numpy")

        # Evaluate at the face centers and scale with face normals
        q_fc = q_fun(fc[1]) * fn[1]

        return q_fc

    def fracture_source(self, sd_frac: pp.Grid) -> np.ndarray:
        """Compute exact integrated fracture source.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact integrated
            sources.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Cell centers and volumes
        cc = sd_frac.cell_centers
        vol = sd_frac.cell_volumes

        # Lambdify expression
        f_fun = sym.lambdify(y, self.f_frac, "numpy")

        # Evaluate and integrate
        f_cc = f_fun(cc[1]) * vol

        return f_cc

    def interface_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        """Compute exact mortar fluxes at the interface.

        Parameters:
            intf: Mortar grid.

        Returns:
            Array of ``shape=(intf.num_cells, )`` containing the exact mortar fluxes.

        Note:
            The returned mortar fluxes are already scaled with ``intf.cell_volumes``.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Cell centers and volumes
        cc = intf.cell_centers
        vol = intf.cell_volumes

        # Lambdify expression
        lmbda_fun = sym.lambdify(y, self.q_intf, "numpy")

        # Evaluate and integrate
        lmbda_cc = lmbda_fun(cc[1]) * vol

        return lmbda_cc

    def boundary_values(self, sd_rock: pp.Grid) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            sd_rock: Rock grid.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` with the exact pressure values
            at the exterior boundary faces.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of face indices
        fc = sd_rock.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Boundary faces
        bc_faces = sd_rock.get_boundary_faces()

        # Lambdify expression
        p_fun = [sym.lambdify((x, y), p, "numpy") for p in self.p_rock]

        # Boundary pressures
        p_bf = np.zeros(sd_rock.num_faces)
        for (p, idx) in zip(p_fun, face_idx):
            p_bf[bc_faces] += p(fc[0], fc[1])[bc_faces] * idx[bc_faces]

        return p_bf


class ModifiedDataSavingMixin(pp.DataSavingMixin):
    """Mixin class to save relevant data."""

    _nonlinear_iteration: int
    """Number of non-linear iterations needed to solve the system. Used only as an
    indicator to avoid saving the initial conditions.

    """

    exact_sol: ManuIncompExactSolution
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
        """Save data to the `results` list."""
        if self._nonlinear_iteration > 0:  # avoid saving the initial condition
            collected_data: SaveData = self._collect_data()
            self.results.append(collected_data)

    def _collect_data(self) -> SaveData:
        """Collect data from the verification setup.

        Returns:
            SaveData object containing the results of the verification.

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
        exact_sol: ManuIncompExactSolution = self.exact_sol

        # Instantiate data class
        out = SaveData()

        # Rock pressure
        out.exact_rock_pressure = exact_sol.rock_pressure(sd_rock)
        out.approx_rock_pressure = data_rock[pp.STATE][p_name]
        out.error_rock_pressure = self.relative_l2_error(
            grid=sd_rock,
            true_array=out.exact_rock_pressure,
            approx_array=out.approx_rock_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Fracture pressure
        out.exact_frac_pressure = exact_sol.fracture_pressure(sd_frac)
        out.approx_frac_pressure = data_frac[pp.STATE][p_name]
        out.error_frac_pressure = self.relative_l2_error(
            grid=sd_frac,
            true_array=out.exact_frac_pressure,
            approx_array=out.approx_frac_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Rock flux
        out.exact_rock_flux = exact_sol.rock_flux(sd_rock)
        out.approx_rock_flux = data_rock[pp.STATE][q_name]
        out.error_rock_flux = self.relative_l2_error(
            grid=sd_rock,
            true_array=out.exact_rock_flux,
            approx_array=out.approx_rock_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Fracture flux
        out.exact_frac_flux = exact_sol.fracture_flux(sd_frac)
        out.approx_frac_flux = data_frac[pp.STATE][q_name]
        out.error_frac_flux = self.relative_l2_error(
            grid=sd_frac,
            true_array=out.exact_frac_flux,
            approx_array=out.approx_frac_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Interface flux
        out.exact_intf_flux = exact_sol.interface_flux(intf)
        out.approx_intf_flux = data_intf[pp.STATE][lmbda_name]
        out.error_intf_flux = self.relative_l2_error(
            grid=intf,
            true_array=out.exact_intf_flux,
            approx_array=out.approx_intf_flux,
            is_scalar=True,
            is_cc=True,
        )

        return out


class SetupUtilities:
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[SaveData]
    """List of SaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_rock_pressure()
        self._plot_fracture_pressure()
        self._plot_interface_fluxes()
        self._plot_fracture_fluxes()

    def _plot_rock_pressure(self) -> None:
        """Plots exact and numerical pressures in the rock."""
        sd_rock = self.mdg.subdomains()[0]
        p_num = self.results[-1].approx_rock_pressure
        p_ex = self.results[-1].exact_rock_pressure
        pp.plot_grid(
            sd_rock, p_ex, plot_2d=True, linewidth=0, title="Rock pressure (Exact)"
        )
        pp.plot_grid(
            sd_rock, p_num, plot_2d=True, linewidth=0, title="Rock pressure (MPFA)"
        )

    def _plot_fracture_pressure(self):
        """Plots exact and numerical pressures in the fracture."""
        sd_frac = self.mdg.subdomains()[1]
        cc = sd_frac.cell_centers
        p_num = self.results[-1].approx_frac_pressure
        p_ex = self.results[-1].exact_frac_pressure
        plt.plot(p_ex, cc[1], label="Exact", linewidth=3, alpha=0.5)
        plt.plot(p_num, cc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
        plt.xlabel("Fracture pressure")
        plt.ylabel("y-coordinate")
        plt.legend()
        plt.show()

    def _plot_interface_fluxes(self):
        """Plots exact and numerical interface fluxes."""
        intf = self.mdg.interfaces()[0]
        cc = intf.cell_centers
        lmbda_num = self.results[-1].approx_intf_flux
        lmbda_ex = self.results[-1].exact_intf_flux
        plt.plot(lmbda_ex, cc[1], label="Exact", linewidth=3, alpha=0.5)
        plt.plot(lmbda_num, cc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
        plt.xlabel("Interface flux")
        plt.ylabel("y-coordinate")
        plt.legend()
        plt.show()

    def _plot_fracture_fluxes(self):
        """Plots exact and numerical fracture fluxes."""
        sd_frac = self.mdg.subdomains()[1]
        fc = sd_frac.face_centers
        q_num = self.results[-1].approx_frac_flux
        q_ex = self.results[-1].exact_frac_flux
        plt.plot(q_ex, fc[1], label="Exact", linewidth=3, alpha=0.5)
        plt.plot(q_num, fc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
        plt.xlabel("Fracture Darcy flux")
        plt.ylabel("y-coordinate")
        plt.legend()
        plt.show()


class SingleEmbeddedVerticalFracture(pp.ModelGeometry):
    """Generate fracture network and mixed-dimensional grid."""

    params: dict
    """Simulation model parameters"""

    fracture_network: pp.FractureNetwork2d
    """Fracture network. Set in :meth:`set_fracture_network`."""

    def set_fracture_network(self) -> None:
        """Create fracture network.

        Note:
            Two horizontal fractures at :math:`y = 0.25` and :math:`y = 0.75` are
            included in the fracture network to force the grid to conform to certain
            regions of the domain. Note, however, that these fractures will not be
            part of the mixed-dimensional grid.

        """
        # Unit square domain
        domain: dict[str, number] = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        # Point coordinates
        point_coordinates = np.array(
            [[0.50, 0.50, 0.00, 1.00, 0.00, 1.00], [0.25, 0.75, 0.25, 0.25, 0.75, 0.75]]
        )

        # Point connections
        point_indices = np.array([[0, 2, 4], [1, 3, 5]])

        # Create fracture network
        network_2d = pp.FractureNetwork2d(point_coordinates, point_indices, domain)
        self.fracture_network = network_2d

    def mesh_arguments(self) -> dict:
        """Define mesh arguments for meshing."""
        return self.params.get(
            "mesh_arguments", {"mesh_size_bound": 0.1, "mesh_size_frac": 0.1}
        )

    def set_md_grid(self) -> None:
        """Create mixed-dimensional grid. Ignore fractures 1 and 2."""
        self.mdg = self.fracture_network.mesh(
            self.mesh_arguments(), constraints=np.array([1, 2])
        )
        domain = self.fracture_network.domain
        # Convert domain to dictionary if it is given as a numpy array.
        # TODO: We should rather unify the way we define the domain in the
        # FractureNetwork2d class.
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


class ManuIncompBoundaryConditions:
    """Set boundary conditions for the simulation model."""

    domain_boundary_sides: Callable[[pp.Grid], pp.bounding_box.DomainSides]
    """Utility function to access the domain boundary sides."""

    exact_sol: ManuIncompExactSolution
    """Exact solution object."""

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Set boundary condition type."""
        if sd.dim == 2:  # Dirichlet for the rock
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        else:  # Neumann for the fracture tips
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Set boundary condition values."""
        values = []
        for sd in subdomains:
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            val_loc = np.zeros(sd.num_faces)
            if sd.dim == 2:
                exact_bc = self.exact_sol.boundary_values(sd_rock=sd)
                val_loc[boundary_faces] = exact_bc[boundary_faces]
            values.append(val_loc)
        return pp.wrap_as_ad_array(np.hstack(values), name="bc_values_darcy")


class ManuIncompBalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    exact_sol: ManuIncompExactSolution
    """Exact solution object."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Contribution of mass fluid sources to the mass balance equation.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise Ad operator containing the fluid source contributions.

        """
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # Retrieve external (integrated) sources from the exact solution.
        values = []
        for sd in subdomains:
            if sd.dim == 2:
                values.append(self.exact_sol.rock_source(sd_rock=sd))
            else:
                values.append(self.exact_sol.fracture_source(sd_frac=sd))
        external_sources = pp.wrap_as_ad_array(np.hstack(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


class ManuIncompSolutionStrategy(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`porepy.models.constitutive_laws.DarcysLaw`.

    """

    exact_sol: ManuIncompExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants
    """Object containing the fluid constants."""

    plot_results: Callable
    """Method to plot results of the verification setup. Usually provided by the
    mixin class :class:`SetupUtilities`.

    """

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    results: list[SaveData]
    """List of SaveData objects."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: ManuIncompExactSolution
        """Exact solution object."""

        self.results: list[SaveData] = []
        """Results object that stores exact and approximated solutions and errors."""

        self.subdomain_darcy_flux_variable: str = "darcy_flux"
        """Keyword to access the subdomain Darcy fluxes."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks to guarantee the validity of the manufactured solution
        assert self.fluid.density() == 1
        assert self.fluid.viscosity() == 1
        assert self.fluid.compressibility() == 0
        assert self.solid.permeability() == 1
        assert self.solid.residual_aperture() == 1
        assert self.solid.normal_permeability() == 0.5

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ManuIncompExactSolution()

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after if solution has converged."""

        # Store subdomain Darcy fluxes in pp.STATE
        for sd, data in self.mdg.subdomains(return_data=True):
            darcy_flux_ad = self.darcy_flux([sd])
            vals = darcy_flux_ad.evaluate(self.equation_system).val
            data[pp.STATE]["darcy_flux"] = vals

        # Inherit from base class
        super().after_nonlinear_convergence(solution, errors, iteration_counter)

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False


class ManufacturedIncompressibleFlow2d(  # type: ignore[misc]
    ManuIncompBalanceEquation,
    ManuIncompBoundaryConditions,
    ManuIncompSolutionStrategy,
    pp.fluid_mass_balance.SinglePhaseFlow,
    SetupUtilities,
    VerificationUtils,
    SingleEmbeddedVerticalFracture,
    ModifiedDataSavingMixin,
):
    """
    Mixer class for the incompressible flow with a single fracture verification setup.

    Examples:

        .. code:: python

            from time import time
            from porepy.applications.verification_setups.manu_flow_incomp_frac import (
                manu_incomp_solid,
                manu_incomp_fluid,
            )

            # Simulation parameters
            material_constants = {
                "solid": pp.SolidConstants(manu_incomp_solid),
                "fluid": pp.FluidConstants(manu_incomp_fluid),
            }

            params = {"material_constants": material_constants, "plot_results": True}

            # Run verification setup
            tic = time()
            setup = ManufacturedIncompressibleFlow2d(params)
            print("Simulation started...")
            pp.run_stationary_model(setup, params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds.")

    """
