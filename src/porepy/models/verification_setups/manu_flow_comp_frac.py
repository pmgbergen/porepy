"""
This module contains a code verification implementation for a manufactured solution for
the two-dimensional, compressible, single phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

The exact solution was obtained by extending the solution from the incompressible
case, see https://doi.org/10.1515/jnma-2022-0038

TODO: Explain what are the modifications w.r.t. the incompressible case.
FIXME: There is significant code duplication with the incompressible verification.
 Consider recycling classes.

"""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import porepy as pp
from porepy.models.verification_setups.verifications_utils import VerificationUtils


class ExactSolution:
    """Parent class for the exact solution."""

    def __init__(self):
        """Constructor of the class."""

        # FIXME: Retrieve the physical constants from the model params dictionary
        # Physical parameters
        rho_0 = 1.0
        p_0 = 0.0
        c_f = 0.2
        phi = 0.1

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

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
        accum_rock = [sym.diff(phi * rho, t) for rho in rho_rock]

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
        accum_frac = sym.diff(phi * rho_frac, t)

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

    # -----> Rock variables
    def rock_pressure(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
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

    def rock_density(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact fluid density [kg * m^-3] in the rock at the cell centers.

        Parameters:
            sd_rock: Rock grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact densities at
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
        rho_fun = [sym.lambdify((x, y, t), rho, "numpy") for rho in self.rho_rock]

        # Cell-centered densities
        rho_cc = np.zeros(sd_rock.num_cells)
        for (rho, idx) in zip(rho_fun, cell_idx):
            rho_cc += rho(cc[0], cc[1], time) * idx

        return rho_cc

    def rock_darcy_flux(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
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

    def rock_mass_flux(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact mass flux [kg * s^-1] at the face centers.

        Parameters:
            sd_rock: Rock grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` containing the exact mass
            fluxes at the face centers for the given ``time``.

        Note:
            The returned mass fluxes are already scaled with ``sd_rock.face_normals``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd_rock.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify expression
        mf_fun = [
            [
                sym.lambdify((x, y, t), mf[0], "numpy"),
                sym.lambdify((x, y, t), mf[1], "numpy"),
            ]
            for mf in self.mf_rock
        ]

        # Face-centered mass fluxes
        fn = sd_rock.face_normals
        mf_fc = np.zeros(sd_rock.num_faces)
        for (mf, idx) in zip(mf_fun, face_idx):
            mf_fc += (
                mf[0](fc[0], fc[1], time) * fn[0] + mf[1](fc[0], fc[1], time) * fn[1]
            ) * idx

        # Again, we need to correct the values at the internal boundaries
        # Here, we simply use the fact that mf_rock := rho_rock * q_rock = 1 * q_rock
        # on the internal boundaries
        q_fc = self.rock_darcy_flux(sd_rock, time)
        frac_faces = sd_rock.tags["fracture_faces"]
        mf_fc[frac_faces] = q_fc[frac_faces]

        return mf_fc

    # TODO: Implement method
    def rock_accumulation(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        raise NotImplementedError

    def rock_source(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
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

    # ----> Fracture variables
    def fracture_pressure(self, sd_frac: pp.Grid, time: float) -> np.ndarray:
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

    def fracture_density(self, sd_frac: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact fracture density at the cell centers.

        Parameters:
            sd_frac: Fracture grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact densities at
            the cell centers at the given ``time``.

        """
        # Symbolic variable
        y, t = sym.symbols("y t")

        # Cell centers
        cc = sd_frac.cell_centers

        # Lambdify expression
        rho_fun = sym.lambdify((y, t), self.rho_frac, "numpy")

        # Evaluate at the cell centers
        rho_cc = rho_fun(cc[1], time)

        return rho_cc

    def fracture_darcy_flux(self, sd_frac: pp.Grid, time: float) -> np.ndarray:
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

    # TODO: Implement method
    def fracture_mass_flux(self, sd_frac: pp.Grid, time: float) -> np.ndarray:
        raise NotImplementedError

    # TODO: Implement method
    def fracture_accumulation(self, sd_frac: pp.Grid, time: float) -> np.ndarray:
        raise NotImplementedError

    def fracture_source(self, sd_frac: pp.Grid, time: float) -> np.ndarray:
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

    # -----> Interface variables
    def interface_darcy_flux(self, intf: pp.MortarGrid, time: float) -> np.ndarray:
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

    # TODO: Implement method
    def interface_mass_flux(self, intf: pp.MortarGrid, time: float) -> np.ndarray:
        raise NotImplementedError

    # -----> Others
    def rock_boundary_pressure(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
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


class StoreResults(VerificationUtils):
    """Class to store results."""

    def __init__(self, setup):
        """Constructor of the class"""

        # Retrieve information from setup
        mdg = setup.mdg
        sd_rock = mdg.subdomains()[0]
        data_rock = mdg.subdomain_data(sd_rock)
        sd_frac = mdg.subdomains()[1]
        data_frac = mdg.subdomain_data(sd_frac)
        intf = mdg.interfaces()[0]
        data_intf = mdg.interface_data(intf)
        p_name = setup.pressure_variable
        q_name = "darcy_flux"
        lmbda_name = setup.interface_darcy_flux_variable
        t = setup.time_manager.time

        # Instantiate exact solution object
        # FIXME: Access exact solution via `setup`
        ex = ExactSolution()

        # Store time for convenience
        self.time = t

        # Rock pressure
        self.exact_rock_pressure = ex.rock_pressure(sd_rock, t)
        self.approx_rock_pressure = data_rock[pp.STATE][pp.ITERATE][p_name]
        self.error_rock_pressure = self.relative_l2_error(
            grid=sd_rock,
            true_array=self.exact_rock_pressure,
            approx_array=self.approx_rock_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Rock flux
        self.exact_rock_flux = ex.rock_darcy_flux(sd_rock, t)
        self.approx_rock_flux = data_rock[pp.STATE][pp.ITERATE][q_name]
        self.error_rock_flux = self.relative_l2_error(
            grid=sd_rock,
            true_array=self.exact_rock_flux,
            approx_array=self.approx_rock_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Fracture pressure
        self.exact_frac_pressure = ex.fracture_pressure(sd_frac, t)
        self.approx_frac_pressure = data_frac[pp.STATE][pp.ITERATE][p_name]
        self.error_frac_pressure = self.relative_l2_error(
            grid=sd_frac,
            true_array=self.exact_frac_pressure,
            approx_array=self.approx_frac_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Fracture flux
        self.exact_frac_flux = ex.fracture_darcy_flux(sd_frac, t)
        self.approx_frac_flux = data_frac[pp.STATE][pp.ITERATE][q_name]
        self.error_frac_flux = self.relative_l2_error(
            grid=sd_frac,
            true_array=self.exact_frac_flux,
            approx_array=self.approx_frac_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Mortar flux
        self.exact_intf_flux = ex.interface_darcy_flux(intf, t)
        self.approx_intf_flux = data_intf[pp.STATE][pp.ITERATE][lmbda_name]
        self.error_intf_flux = self.relative_l2_error(
            grid=intf,
            true_array=self.exact_intf_flux,
            approx_array=self.approx_intf_flux,
            is_scalar=True,
            is_cc=True,
        )


# -----> Simulation model
# FIXME: Resolve duplication with the incompressible verification
class ModifiedGeometry(pp.ModelGeometry):
    """Generate fracture network and mixed-dimensional grid."""

    params: dict
    """Simulation model parameters"""

    def set_fracture_network(self) -> None:
        """Create fracture network.

        Note:
            Two horizontal fractures at y = 0.25 and y = 0.75 are included in the
            fracture network to force the grid to conform to certain regions of the
            domain. Note, however, that these fractures will not be part of the
            mixed-dimensional grid.

        """
        # Unit square domain
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

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
        default_mesh_arguments = {"mesh_size_bound": 0.05, "mesh_size_frac": 0.05}
        return self.params.get("mesh_arguments", default_mesh_arguments)

    def set_md_grid(self) -> None:
        """Create mixed-dimensional grid. Ignore fractures 1 and 2."""
        self.mdg = self.fracture_network.mesh(
            self.mesh_arguments(), constraints=np.array([1, 2])
        )
        assert isinstance(self.fracture_network.domain, dict)
        self.domain_bounds = self.fracture_network.domain  # type: ignore


class ModifiedBoundaryConditions:
    """Set boundary conditions for the simulation model."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid object."""

    time_manager: pp.TimeManager
    """Properly initialized time manager object."""

    domain_boundary_sides: Callable[
        [pp.Grid],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
    """Utility function to access the domain boundary sides."""

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:  # Dirichlet for the rock
            all_bf, *_ = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all_bf, "dir")
        else:  # Neumann for the fracture tips
            all_bf, *_ = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all_bf, "neu")

    def bc_type_mobrho(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:  # Dirichlet for the rock
            all_bf, *_ = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all_bf, "dir")
        else:  # Neumann for the fracture tips
            all_bf, *_ = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all_bf, "neu")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        bc_values = pp.ad.TimeDependentArray(
            name="darcy_bc_values", subdomains=subdomains, previous_timestep=True
        )
        return bc_values

    def bc_values_mobrho(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        bc_values = pp.ad.TimeDependentArray(
            name="mobrho_bc_values", subdomains=subdomains, previous_timestep=True
        )
        return bc_values


class ModifiedBalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:

        # Internal sources are inherit from parent class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # External sources are retrieved from STATE and wrapped as an AdArray
        external_sources = pp.ad.TimeDependentArray(
            name="external_sources",
            subdomains=self.mdg.subdomains(),
            previous_timestep=True,
        )

        return internal_sources + external_sources


class ModifiedSolutionStrategy(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):

    fluid: pp.FluidConstants
    mdg: pp.MixedDimensionalGrid
    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    results: list[StoreResults]

    def __init__(self, params: dict):

        # Parameters associated with the verification setup. The below parameters
        # cannot be changed since they're associated with the exact solution
        fluid = pp.FluidConstants({"compressibility": 0.2})
        solid = pp.SolidConstants({"porosity": 0.1, "residual_aperture": 1})
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
        """Update values of external sources and boundary conditions"""

        # Retrieve subdomains and data dictionaries
        sd_rock = self.mdg.subdomains()[0]
        data_rock = self.mdg.subdomain_data(sd_rock)
        sd_frac = self.mdg.subdomains()[1]
        data_frac = self.mdg.subdomain_data(sd_frac)

        # Retrieve current time
        t = self.time_manager.time

        # -----> Sources
        rock_source = self.exact_sol.rock_source(sd_rock=sd_rock, time=t)
        data_rock[pp.STATE]["external_sources"] = rock_source

        frac_source = self.exact_sol.fracture_source(sd_frac=sd_frac, time=t)
        data_frac[pp.STATE]["external_sources"] = frac_source

        # -----> Boundary conditions

        # Boundary conditions for the elliptic discretization
        rock_pressure_boundary = self.exact_sol.rock_boundary_pressure(
            sd_rock=sd_rock, time=t
        )
        data_rock[pp.STATE]["darcy_bc_values"] = rock_pressure_boundary
        data_frac[pp.STATE]["darcy_bc_values"] = np.zeros(sd_frac.num_faces)

        # Boundary conditions for the upwind discretization
        rock_density_boundary = self.exact_sol.rock_boundary_density(
            sd_rock=sd_rock, time=t
        )
        viscosity = self.fluid.viscosity()
        rock_mobrho = rock_density_boundary / viscosity
        data_rock[pp.STATE]["mobrho_bc_values"] = rock_mobrho
        data_frac[pp.STATE]["mobrho_bc_values"] = np.zeros(sd_frac.num_faces)

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after convergence"""

        # FIXME: Retrieve Darcy fluxes using the mobility keyword
        # Retrieve Darcy fluxes and store them in pp.STATE
        darcy_flux = self.darcy_flux(self.mdg.subdomains())
        vals = darcy_flux.evaluate(self.equation_system).val
        nf_rock = self.mdg.subdomains()[0].num_faces
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == 2:
                data[pp.STATE][pp.ITERATE]["darcy_flux"] = vals[:nf_rock]
            else:
                data[pp.STATE][pp.ITERATE]["darcy_flux"] = vals[nf_rock:]

        # Store results
        t = self.time_manager.time
        if np.any(np.isclose(t, self.params["stored_times"])):
            self.results.append(StoreResults(self))

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""

        if self.params.get("plot_results", False):
            self.plot_results()

    # -----> Plot-related methods
    def plot_results(self) -> None:
        """Plotting results"""

        self._plot_rock_pressure()
        self._plot_fracture_pressure()
        self._plot_interface_fluxes()
        self._plot_fracture_fluxes()

    def _plot_rock_pressure(self):
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

    # FIXME: Use sidegrids to separate left and right mortar fluxes
    def _plot_interface_fluxes(self):
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


class ManufacturedCompFlow2d(  # type: ignore[misc]
    ModifiedGeometry,
    ModifiedBoundaryConditions,
    ModifiedBalanceEquation,
    ModifiedSolutionStrategy,
    pp.fluid_mass_balance.SinglePhaseFlow,
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
            setup = ManufacturedCompFlow2d(params)
            print("Simulation started...")
            pp.run_time_dependent_model(setup, params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds.")
    """

    ...
