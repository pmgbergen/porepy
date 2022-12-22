"""
This module contains a code verification implementation for a manufactured solution for
the two-dimensional, incompressible, single phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

For the details, we refer to https://doi.org/10.1515/jnma-2022-0038

TODO: The placement of this module is not ideal. It will be moved to a more appropriate
location in the future.

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

        # Exact mortar flux
        lmbda = bubble_fun

        # Exact pressure in the fracture
        p_frac = -bubble_fun

        # Exact Darcy flux in the fracture
        q_frac = -sym.diff(p_frac, y)

        # Exact source in the fracture
        f_frac = sym.diff(q_frac, y) - 2 * lmbda

        # Public attributes
        self.p_rock = p_rock
        self.q_rock = q_rock
        self.f_rock = f_rock
        self.p_frac = p_frac
        self.q_frac = q_frac
        self.f_frac = f_frac
        self.lmbda = lmbda

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

    def mortar_fluxes(self, intf: pp.MortarGrid) -> np.ndarray:
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
        lmbda_fun = sym.lambdify(y, self.lmbda, "numpy")

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

        # Instantiate exact solution object
        ex = ExactSolution()

        # Rock pressure
        self.exact_rock_pressure = ex.rock_pressure(sd_rock)
        self.approx_rock_pressure = data_rock[pp.STATE][p_name]
        self.error_rock_pressure = self.relative_l2_error(
            grid=sd_rock,
            true_array=self.exact_rock_pressure,
            approx_array=self.approx_rock_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Rock flux
        self.exact_rock_flux = ex.rock_flux(sd_rock)
        self.approx_rock_flux = data_rock[pp.STATE][q_name]
        self.error_rock_flux = self.relative_l2_error(
            grid=sd_rock,
            true_array=self.exact_rock_flux,
            approx_array=self.approx_rock_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Fracture pressure
        self.exact_frac_pressure = ex.fracture_pressure(sd_frac)
        self.approx_frac_pressure = data_frac[pp.STATE][p_name]
        self.error_frac_pressure = self.relative_l2_error(
            grid=sd_frac,
            true_array=self.exact_frac_pressure,
            approx_array=self.approx_frac_pressure,
            is_scalar=True,
            is_cc=True,
        )

        # Fracture flux
        self.exact_frac_flux = ex.fracture_flux(sd_frac)
        self.approx_frac_flux = data_frac[pp.STATE][q_name]
        self.error_frac_flux = self.relative_l2_error(
            grid=sd_frac,
            true_array=self.exact_frac_flux,
            approx_array=self.approx_frac_flux,
            is_scalar=True,
            is_cc=False,
        )

        # Mortar flux
        self.exact_intf_flux = ex.mortar_fluxes(intf)
        self.approx_intf_flux = data_intf[pp.STATE][lmbda_name]
        self.error_intf_flux = self.relative_l2_error(
            grid=intf,
            true_array=self.exact_intf_flux,
            approx_array=self.approx_intf_flux,
            is_scalar=True,
            is_cc=True,
        )


# -----> Simulation model
class ModifiedGeometry(pp.ModelGeometry):
    """Generate fracture network and mixed-dimensional grid."""

    params: dict
    """Simulation model parameters"""

    fracture_network: pp.FractureNetwork2d
    """Fracture network. Set in :meth:`set_fracture_network`."""

    def set_fracture_network(self) -> None:
        """Create fracture network.

        Note:
            Two horizontal fractures at y = 0.25 and y = 0.75 are included in the
            fracture network to force the grid to conform to certain regions of the
            domain. Note, however, that these fractures will not be part of the
            mixed-dimensional grid.

        """
        # Unit square domain
        domain: dict[str, float] = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

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
        domain = self.fracture_network.domain
        # Convert domain to dictionary if it is given as a numpy array.
        # TODO: We should rather unify the way we define the domain in the
        # FractureNetwork2d class.
        if isinstance(domain, np.ndarray):
            assert domain.shape == (2, 2)
            self.domain: dict[str, float] = {
                "xmin": domain[0, 0],
                "xmax": domain[1, 0],
                "ymin": domain[0, 1],
                "ymax": domain[1, 1],
            }
        else:
            assert isinstance(domain, dict)
            self.domain = domain


class ModifiedBoundaryConditions:
    """Set boundary conditions for the simulation model."""

    domain_boundary_sides: Callable[
        [pp.Grid],
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ]
    """Utility function to access the domain boundary sides."""

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:
            # Define boundary regions
            all_bf, *_ = self.domain_boundary_sides(sd)
            # Define boundary condition on faces
            return pp.BoundaryCondition(sd, all_bf, "dir")
        else:
            # Define boundary regions
            all_bf, *_ = self.domain_boundary_sides(sd)
            # Define boundary condition on faces
            return pp.BoundaryCondition(sd, all_bf, "neu")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        # Define boundary regions
        values = []
        for sd in subdomains:
            all_bf, *_ = self.domain_boundary_sides(sd)
            val_loc = np.zeros(sd.num_faces)
            # See section on scaling for explanation of the conversion.
            if sd.dim == 2:
                ex = ExactSolution()
                val_loc[all_bf] = ex.boundary_values(sd_rock=sd)[all_bf]
            values.append(val_loc)
        return pp.wrap_as_ad_array(np.hstack(values), name="bc_values_darcy")


class ModifiedBalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        ex = ExactSolution()
        values = []
        for sd in subdomains:
            if sd.dim == 2:
                values.append(ex.rock_source(sd_rock=sd))
            else:
                values.append(ex.fracture_source(sd_frac=sd))
        external_sources = pp.wrap_as_ad_array(np.hstack(values))

        return internal_sources + external_sources


class ModifiedSolutionStrategy(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):

    mdg: pp.MixedDimensionalGrid
    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    sol: StoreResults

    def __init__(self, params: dict):

        # Parameters associated with the verification setup. The below parameters
        # cannot be changed since they're associated with the exact solution
        fluid = pp.FluidConstants({"compressibility": 0})
        solid = pp.SolidConstants({"porosity": 0, "residual_aperture": 1})
        material_constants = {"fluid": fluid, "solid": solid}
        required_params = {"material_constants": material_constants}
        params.update(required_params)
        super().__init__(params)

        self.solution: StoreResults
        """Solution object that stores exact and approximated solutions and errors"""

    def plot_results(self) -> None:
        """Plotting results"""

        self._plot_rock_pressure()
        self._plot_fracture_pressure()
        self._plot_interface_fluxes()
        self._plot_fracture_fluxes()

    def _plot_rock_pressure(self):
        sd_rock = self.mdg.subdomains()[0]
        p_num = self.sol.approx_rock_pressure
        p_ex = self.sol.exact_rock_pressure
        pp.plot_grid(
            sd_rock, p_ex, plot_2d=True, linewidth=0, title="Rock pressure (Exact)"
        )
        pp.plot_grid(
            sd_rock, p_num, plot_2d=True, linewidth=0, title="Rock pressure (MPFA)"
        )

    def _plot_fracture_pressure(self):
        sd_frac = self.mdg.subdomains()[1]
        cc = sd_frac.cell_centers
        p_num = self.sol.approx_frac_pressure
        p_ex = self.sol.exact_frac_pressure
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
        lmbda_num = self.sol.approx_intf_flux
        lmbda_ex = self.sol.exact_intf_flux
        plt.plot(lmbda_ex, cc[1], label="Exact", linewidth=3, alpha=0.5)
        plt.plot(lmbda_num, cc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
        plt.xlabel("Interface flux")
        plt.ylabel("y-coordinate")
        plt.legend()
        plt.show()

    def _plot_fracture_fluxes(self):
        sd_frac = self.mdg.subdomains()[1]
        fc = sd_frac.face_centers
        q_num = self.sol.approx_frac_flux
        q_ex = self.sol.exact_frac_flux
        plt.plot(q_ex, fc[1], label="Exact", linewidth=3, alpha=0.5)
        plt.plot(q_num, fc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
        plt.xlabel("Fracture Darcy flux")
        plt.ylabel("y-coordinate")
        plt.legend()
        plt.show()

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished"""

        # TODO: This is ugly. Is there a better way to do this?
        # Compute Darcy fluxes and store them in pp.STATE
        darcy_flux = self.darcy_flux(self.mdg.subdomains())
        vals = darcy_flux.evaluate(self.equation_system).val
        nf_rock = self.mdg.subdomains()[0].num_faces
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == 2:
                data[pp.STATE]["darcy_flux"] = vals[:nf_rock]
            else:
                data[pp.STATE]["darcy_flux"] = vals[nf_rock:]

        # Store solution
        self.sol = StoreResults(setup=self)

        # Plot solution
        plot_sol = self.params.get("plot_results", False)
        if plot_sol:
            self.plot_results()


class ManufacturedFlow2d(  # type: ignore[misc]
    ModifiedGeometry,
    ModifiedBoundaryConditions,
    ModifiedBalanceEquation,
    ModifiedSolutionStrategy,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """
    Mixer class for the manufactured solution with a single vertical fracture.

    Examples:

        .. code:: python

            # Import modules
            import porepy as pp
            from time import time

            # Run verification setup
            tic = time()
            params = {"plot_results": True}
            setup = ManufacturedFlow2d(params)
            print("Simulation started...")
            pp.run_stationary_model(setup, params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds.")
    """

    ...
