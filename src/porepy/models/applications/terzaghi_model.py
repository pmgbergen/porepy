"""
Implementation of Terzaghi's consolidation problem.

Terzaghi's problem is a well known one-dimensional poroelastic problem [1 - 3]. Generally,
when soils are subjected to a vertical load, porosity decreases, resulting in less available
space for pore water. The liquid within the pores can be expelled, however, in certain types
of soils (especially clayey soils) this process may take some time due to their low
permeability. This process is referred to as consolidation.

We consider a soil column of height `h`, where a constant load `F` is applied to the top of
the column while keeping the bottom impervious to flow. The exerted load will cause an
instantaneous rise in the fluid pressure, which will be equal to the applied load. After
that, the fluid pressure will monotonically decrease towards zero.

Even though Terzaghi's consolidation problem is strictly speaking one-dimensional, the
implemented setup employs a two-dimensional Cartesian grid with roller boundary conditions
for the mechanical subproblem and no-flux boundary conditions for the flow subproblem on
the sides of the domain such that the one-dimensional process can be emulated.

The reason why we need to employ a two-dimensional grid is because PorePy only supports
Neumann boundary conditions for the discretization of the elasticity equations in
one-dimensional subdomains.

References:

    [1] von Terzaghi, K. (1923). Die berechnung der durchassigkeitsziffer des tones aus dem
    verlauf der hydrodynamischen spannungs. erscheinungen. Sitzungsber. Akad. Wiss. Math.
    Naturwiss. Kl. Abt. 2A, 132, 105-124.

    [2] von Terzaghi, K. (1944). Theoretical Soil Mechanics.

    [3] Verruijt, A. (2017). An Introduction to Soil Mechanics (Vol. 30). Springer.

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import matplotlib.colors as mcolors  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

import porepy as pp


@dataclass
class TerzaghiSolution:
    """Data class to store variables of interest from Terzaghi's model."""

    def __init__(self, setup: "Terzaghi"):
        """Data class constructor.

        Args:
            setup : Terzaghi model setup.

        """
        sd = setup.mdg.subdomains()[0]
        data = setup.mdg.subdomain_data(sd)
        p_var = setup.scalar_variable
        u_var = setup.displacement_variable
        t = setup.time_manager.time

        # Time variables
        self.time = t
        self.nondim_time = setup.nondim_time(t)

        # Spatial variables
        self.vertical_coo = sd.cell_centers[1]
        self.nondim_vertical_coo = setup.nondim_length(self.vertical_coo)

        # Pressure variables
        self.numerical_pressure = data[pp.STATE][p_var]
        self.exact_pressure = setup.exact_pressure(y=self.vertical_coo, t=self.time)
        self.numerical_nondim_pressure = setup.nondim_pressure(self.numerical_pressure)
        self.exact_nondim_pressure = setup.nondim_pressure(self.exact_pressure)

        # Mechanical variables
        self.numerical_displacement = data[pp.STATE][u_var]
        self.numerical_consolidation_degree = setup.numerical_consolidation_degree(
            displacement=self.numerical_displacement, pressure=self.numerical_pressure
        )
        self.exact_consolidation_degree = setup.exact_consolidation_degree(t)

        # Error variables
        self.pressure_error = setup.l2_relative_error(
            sd=sd,
            true_val=self.exact_pressure,
            approx_val=self.numerical_pressure,
            is_scalar=True,
            is_cc=True,
        )
        self.consolidation_degree_error = np.abs(
            self.exact_consolidation_degree - self.numerical_consolidation_degree
        )


class Terzaghi(pp.ContactMechanicsBiot):
    """Parent class for Terzaghi's setup.

    Examples:

        .. code:: python

            # Import modules
            import porepy as pp
            from time import time

            # Run Terzaghi's setup
            tic = time()
            print("Simulation started...")
            params = {"plot_results": True}
            setup = Terzaghi(params)
            pp.run_time_dependent_model(setup, params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds.")

    """

    def __init__(self, params: dict):
        """Constructor of the Terzaghi class.

        Parameters:
            params: Model setup parameters.

                Default physical parameters were adapted from
                https://link.springer.com/article/10.1007/s10596-013-9393-8.

                Optional parameters are:

                - 'alpha_biot' : Biot constant (int or float). Default is 1.0.
                - 'height' : Height of the domain in `m` (int or float). Default is 1.0.
                - 'lambda_lame' : Lamé parameter in `Pa` (int or float). Defualt is 1.65e9.
                - 'mu_lame' : Lamé parameter in `m` (int or float). Default is 1.475E9.
                - 'num_cells' : Number of vertical cells (int). Default is 20.
                - 'permeability' : Permeability in `m^2` (int or float). Default is 9.86e-14.
                - 'pertubation_factor' : Perturbation factor (int or float). Used for
                  perturbing the physical nodes of the mesh. This is necessary to avoid
                  singular matrices with MPSA and the use of rollers. Default is 1e-6.
                - 'plot_results' : Whether to plot the results (bool). The resulting plot is
                  saved inside the `out` folder. Default is False.
                - 'specific_weight' : Fluid specific weight in `Pa * m^-1` (int or float).
                  Recall that the specific weight is density * gravity. Default is 9.943e3.
                - 'time_manager' : Time manager object (pp.TimeManager). Default is
                  pp.TimeManager([0, 0.01, 0.1, 0.5, 1, 2], 0.001, constant_dt=True).
                - 'use_ad' : Whether to use ad (bool). Must be set to True. Otherwise,
                  an error will be raised. Default is True.
                - 'vertical_load' : Applied vertical load in `N * m^-1` (int or float).
                  Default is 6e8.
                - 'viscosity' : Fluid viscosity in `Pa * s` (int or float). Default is 1e-3.

        """

        def set_default_params(keyword: str, value: object) -> None:
            """
            Set default parameters if a keyword is absent in the `params` dictionary.

            Args:
                keyword: Parameter keyword, e.g., "alpha_biot".
                value: Value of `keyword`, e.g., 1.0.

            """
            if keyword not in params.keys():
                params[keyword] = value

        # Default parameters
        default_tm = pp.TimeManager([0, 0.01, 0.1, 0.5, 1, 2], 0.001, constant_dt=True)
        default_params: list[tuple] = [
            ("alpha_biot", 1.0),  # [-]
            ("height", 1.0),  # [m]
            ("lambda_lame", 1.65e9),  # [Pa]
            ("mu_lame", 1.475e9),  # [Pa]
            ("num_cells", 20),
            ("permeability", 9.86e-14),  # [m^2]
            ("perturbation_factor", 1e-6),
            ("plot_results", False),
            ("specific_weight", 9.943e3),  # [Pa * m^-1]
            ("time_manager", default_tm),  # all time-related variables must be in [s]
            ("upper_limit_summation", 1000),
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
        self.solutions: list[TerzaghiSolution] = []

    def create_grid(self) -> None:
        """Create a two-dimensional Cartesian grid."""

        # Create a standard Cartesian grid
        n = self.params["num_cells"]
        h = self.params["height"]
        phys_dims = np.array([h, h])
        n_cells = np.array([1, n])
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        sd.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])

        # Perturb physical nodes to avoid singular matrices with roller bc and MPSA.
        # Here, we only perturb the vertical nodes, although nothing stop us from perturbing
        # the horizontal nodes too.
        np.random.seed(35)  # this seed is fixed but completely arbitrary
        perturbation_factor = self.params["perturbation_factor"]
        perturbation = np.random.rand(sd.num_nodes) * perturbation_factor
        sd.nodes[1] += perturbation
        sd.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])

    def _initial_condition(self) -> None:
        """Set initial condition.

        Terzaghi's problem assumes that the soil is initially unconsolidated and that
        initial fluid pressure equals the vertical load.

        """
        super()._initial_condition()

        # Since the parent class sets zero initial displacement, we only need to modify the
        # initial conditions for the flow subproblem.
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        vertical_load = self.params["vertical_load"]
        initial_p = vertical_load * np.ones(sd.num_cells)
        data[pp.STATE][self.scalar_variable] = initial_p
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = initial_p

        # Store initial solution
        self.solutions.append(TerzaghiSolution(self))

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            Scalar boundary condition representation.

        """
        # Define boundary regions
        tol = self.params["perturbation_factor"]
        sides = self._domain_boundary_sides(sd, tol)
        north_bc = np.isin(sides.all_bf, np.where(sides.north)).nonzero()

        # All sides Neumann, except the North which is Dirichlet
        bc_type = np.asarray(sides.all_bf.size * ["neu"])
        bc_type[north_bc] = "dir"

        bc = pp.BoundaryCondition(sd, faces=sides.all_bf, cond=list(bc_type))

        return bc

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define boundary condition types for the mechanics subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition representation.

        """
        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        super()._bc_type_mechanics(sd=sd)

        # Get boundary sides, retrieve data dict, and bc object
        tol = self.params["perturbation_factor"]
        sides = self._domain_boundary_sides(sd, tol)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Roller
        bc.is_neu[1, sides.east] = True
        bc.is_dir[1, sides.east] = False

        # West side: Roller
        bc.is_neu[1, sides.west] = True
        bc.is_dir[1, sides.west] = False

        # North side: Neumann
        bc.is_neu[:, sides.north] = True
        bc.is_dir[:, sides.north] = False

        # South side: Dirichlet (already set thanks to inheritance)

        return bc

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition values of shape (sd.dim * sd.num_faces, ).

        """

        # Retrieve boundary sides
        tol = self.params["perturbation_factor"]
        sides = self._domain_boundary_sides(sd, tol)

        # All zeros except vertical component of the north side
        vertical_load = self.params["vertical_load"]
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, sides.north] = -vertical_load * sd.face_areas[sides.north]
        bc_values = bc_values.ravel("F")

        return bc_values

    def after_newton_convergence(
        self,
        solution: np.ndarray,
        errors: float,
        iteration_counter: int,
    ) -> None:
        """Method to be called after the Newton solver has converged."""
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Store solutions
        schedule = self.time_manager.schedule
        if any([np.isclose(self.time_manager.time, t_sch) for t_sch in schedule]):
            self.solutions.append(TerzaghiSolution(self))

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params["plot_results"]:
            self.plot_results()

    # -----> Physical parameters
    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Override value of intrinsic permeability [m^2].

        Args:
            sd: Subdomain grid.

        Returns:
            Array containing the permeability values on each cell.

        """
        return self.params["permeability"] * np.ones(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Override value of storativity [Pa^-1].

        Args:
            sd: Subdomain grid.

        Returns:
            Array containing the storativity values on each cell.

        """
        return np.zeros(sd.num_cells)

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Override stiffness tensor.

        Args:
            sd: Subdomain grid.

        Returns:
            Fourth order tensorial representation of the stiffness tensor.

        """
        lam = (self.params["lambda_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        mu = (self.params["mu_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        return pp.FourthOrderTensor(mu, lam)

    def _viscosity(self, sd: pp.Grid) -> np.ndarray:
        """Override fluid viscosity values [Pa * s].

        Args:
            sd: Subdomain grid.

        Returns:
            Array containing the viscosity values on each cell.

        """
        return self.params["viscosity"] * np.ones(sd.num_cells)

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Override value of Biot-Willis coefficient.

        Args:
            sd: Subdomain grid.

        Returns:
            Array containing the Biot's coefficient on each cell.

        """
        return self.params["alpha_biot"] * np.ones(sd.num_cells)

    def confined_compressibility(self) -> Union[int, float]:
        """Compute confined compressibility [Pa^-1].

        Returns:
            Confined compressibility.

        """
        mu_s = self.params["mu_lame"]
        lambda_s = self.params["lambda_lame"]
        m_v = 1 / (2 * mu_s + lambda_s)

        return m_v

    def consolidation_coefficient(self) -> Union[int, float]:
        """Compute consolidation coefficient [m^2 * s^-1].

        Returns:
            Coefficient of consolidation.

        """
        k = self.params["permeability"]  # [m^2]
        mu_f = self.params["viscosity"]  # [Pa * s]
        gamma_f = self.params["specific_weight"]  # [Pa * m^-1]
        hydraulic_conductivity = (k * gamma_f) / mu_f  # [m * s^-1]

        storativity = 0  # [Pa^-1]
        alpha_biot = self.params["alpha_biot"]  # [-]
        m_v = self.confined_compressibility()  # [Pa^-1]

        c_v = hydraulic_conductivity / (gamma_f * (storativity + alpha_biot**2 * m_v))

        return c_v

    # -----> Exact, numerical, and non-dimensional expressions
    def nondim_time(self, t: Union[float, int]) -> float:
        """Nondimensionalize time.

        Args:
            t: Time in seconds.

        Returns:
            Dimensionless time for the given time `t`.

        """
        h = self.params["height"]
        c_v = self.consolidation_coefficient()

        return (t * c_v) / (h**2)

    def nondim_length(
        self, length: Union[float, int, np.ndarray]
    ) -> Union[float, int, np.ndarray]:
        """Nondimensionalize length.

        Args:
            length : length in meters.

        Returns:
            Non-dimensionalized length.

        """
        return length / self.params["height"]

    def nondim_pressure(self, pressure: np.ndarray) -> np.ndarray:
        """Nondimensionalize pressure.

        Args:
            pressure : pressure in Pa.

        Returns:
            Non-dimensional pressure.

        """
        return pressure / np.abs(self.params["vertical_load"])

    def exact_pressure(self, y: np.ndarray, t: Union[float, int]) -> np.ndarray:
        """Compute exact pressure.

        Args:
            y: vertical coordinates in meters.
            t: Time in seconds.

        Returns:
            Exact pressure profile for the given time ``t``.

        """
        F = self.params["vertical_load"]
        nondim_y = self.nondim_length(y)
        nondim_t = self.nondim_time(t)

        n = self.params["upper_limit_summation"]

        if t == 0:  # initally, the pressure equals the vertical load
            p = F * np.ones_like(y)
        else:
            sum_series = np.zeros_like(y)
            for i in range(1, n + 1):
                sum_series += (
                    (((-1) ** (i - 1)) / (2 * i - 1))
                    * np.cos((2 * i - 1) * (np.pi / 2) * nondim_y)
                    * np.exp((-((2 * i - 1) ** 2)) * (np.pi**2 / 4) * nondim_t)
                )
            p = (4 / np.pi) * F * sum_series

        return p

    def exact_consolidation_degree(self, t: Union[float, int]) -> float:
        """Compute exact degree of consolidation.

        Args:
            t : time in seconds.

        Returns:
            Degree of consolidation for the given time.

        """
        t_nondim = self.nondim_time(t)

        if t == 0:  # initially, the soil is unconsolidated
            deg_cons = 0.0
        else:
            sum_series = 0
            for i in range(1, self.params["upper_limit_summation"] + 1):
                sum_series += (
                    1
                    / ((2 * i - 1) ** 2)
                    * np.exp(-((2 * i - 1) ** 2) * (np.pi**2 / 4) * t_nondim)
                )
            deg_cons = 1 - (8 / (np.pi**2)) * sum_series

        return deg_cons

    def numerical_consolidation_degree(
        self, displacement: np.ndarray, pressure: np.ndarray
    ) -> float:
        """Numerical consolidation coefficient.

        Args:
            displacement: Displacement solution of shape (sd.dim * sd.num_cells, ).
            pressure: Pressure solution of shape (sd.num_cells, ).

        Returns:
            Numerical degree of consolidation.

        """
        sd = self.mdg.subdomains()[0]
        h = self.params["height"]
        m_v = self.confined_compressibility()
        vertical_load = self.params["vertical_load"]
        t = self.time_manager.time

        if t == 0:  # initially, the soil is unconsolidated
            consol_deg = 0.0
        else:
            trace_u = self.displacement_trace(displacement, pressure)
            u_inf = m_v * h * vertical_load
            u_0 = 0
            u = np.max(np.abs(trace_u[1 :: sd.dim]))
            consol_deg = (u - u_0) / (u_inf - u_0)

        return consol_deg

    def displacement_trace(
        self, displacement: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Project the displacement vector onto the faces.

        Args:
            displacement: displacement solution of shape (sd.dim * sd.num_cells, ).
            pressure: pressure solution of shape (sd.num_cells, ).

        Returns:
            Trace of the displacement with shape (sd.dim * sd.num_faces, ).

        """
        # Rename arguments
        u = displacement
        p = pressure

        # Discretization matrices
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        discr = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]
        bound_u_cell = discr["bound_displacement_cell"]
        bound_u_face = discr["bound_displacement_face"]
        bound_u_pressure = discr["bound_displacement_pressure"]

        # Mechanical boundary values
        bc_vals = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc_values"]

        # Compute trace of the displacement
        trace_u = bound_u_cell * u + bound_u_face * bc_vals + bound_u_pressure * p

        return trace_u

    # -----> Helper methods
    def plot_results(self) -> None:
        """Plot the results"""
        # Retrieve colormap from the tab20 pallete
        cmap = mcolors.ListedColormap(
            plt.cm.tab20.colors[: len(self.time_manager.schedule)]
        )

        # Pressure plot
        self._pressure_plot(
            folder="out/",
            file_name="nondimensional_pressure",
            file_extension=".pdf",
            color_map=cmap,
        )

        # Degree of consolidation plot
        self._consolidation_degree_plot(
            folder="out/",
            file_name="consolidation_degree",
            file_extension=".pdf",
            color_map=cmap,
        )

    def _pressure_plot(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot nondimensional pressure profiles.

        Args:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "pressure_profiles".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.

        """

        fig, ax = plt.subplots(figsize=(9, 8))

        y_ex = np.linspace(0, self.params["height"], 400)
        for idx, sol in enumerate(self.solutions):
            ax.plot(
                self.nondim_pressure(self.exact_pressure(y=y_ex, t=sol.time)),
                self.nondim_length(y_ex),
                color=color_map.colors[idx],
            )
            ax.plot(
                sol.numerical_nondim_pressure,
                sol.nondim_vertical_coo,
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
                label=rf"$\tau=${np.round(sol.nondim_time, 5)}",
            )

        ax.set_xlabel(r"$\tilde{p} = p/p_0$", fontsize=15)
        ax.set_ylabel(r"$\tilde{y} = y/h$", fontsize=15)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.set_title("Normalized pressure profiles", fontsize=16)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    def _consolidation_degree_plot(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot the degree of consolidation versus non-dimesional time.

        Args:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "pressure_profiles".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.

        """

        # Retrieve data
        t_ex = np.linspace(
            self.time_manager.time_init, self.time_manager.time_final, 400
        )
        nondim_t_ex = np.asarray([self.nondim_time(t) for t in t_ex])
        exact_consolidation = np.asarray(
            [self.exact_consolidation_degree(t) for t in t_ex]
        )

        nondim_t = np.asarray([sol.nondim_time for sol in self.solutions])
        numerical_consolidation = np.asarray(
            [sol.numerical_consolidation_degree for sol in self.solutions]
        )

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.semilogx(
            nondim_t_ex, exact_consolidation, color=color_map.colors[0], label="Exact"
        )
        ax.semilogx(
            nondim_t,
            numerical_consolidation,
            color=color_map.colors[0],
            linewidth=0,
            marker=".",
            markersize=12,
            label="Numerical",
        )
        ax.set_xlabel(r"$\tau(t)$", fontsize=15)
        ax.set_ylabel(r"$U(t)$", fontsize=15)
        ax.legend(fontsize=14)
        ax.set_title("Degree of consolidation vs. non-dimensional time", fontsize=16)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    @staticmethod
    def l2_relative_error(
        sd: pp.Grid,
        true_val: np.ndarray,
        approx_val: np.ndarray,
        is_cc: bool,
        is_scalar: bool,
    ) -> float:
        """Compute the error measured in the discrete (relative) L2-norm.

        The employed norms correspond respectively to equations (75) and (76) for the
        displacement and pressure from https://epubs.siam.org/doi/pdf/10.1137/15M1014280.

        Args:
            sd: PorePy grid.
            true_val: Exact array, e.g.: pressure, displacement, flux, or traction.
            approx_val: Approximated array, e.g.: pressure, displacement, flux, or traction.
            is_cc: True for cell-centered quantities (e.g., pressure and displacement)
                and False for face-centered quantities (e.g., flux and traction).
            is_scalar: True for scalar quantities (e.g., pressure or flux) and False for
                vector quantities (displacement and traction).

        Returns:
            Discrete L2-error of the quantity of interest.

        """
        if is_cc:
            if is_scalar:
                meas = sd.cell_volumes
            else:
                meas = sd.cell_volumes.repeat(sd.dim)
        else:
            if is_scalar:
                meas = sd.cell_faces
            else:
                meas = sd.cell_faces.repeat(sd.dim)

        numerator = np.sqrt(np.sum(meas * np.abs(true_val - approx_val) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_val) ** 2))
        l2_error = numerator / denominator

        return l2_error
