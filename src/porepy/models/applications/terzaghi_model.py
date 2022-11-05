"""
Implementation of Terzaghi's consolidation problem.

Even though Terzaghi's consolidation problem is strictly speaking one-dimensional, the
current model employs a two-dimensional Cartesian grid with roller boundary conditions
for the mechanical subproblem and no-flux boundary conditions for the flow subproblem on
the sides of the domain such that the one-dimensional process can be emulated.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import porepy as pp
import os

from typing import Union


class Terzaghi(pp.ContactMechanicsBiot):
    """Parent class for Terzaghi's consolidation problem model.

    Examples:

        .. code:: Python

        # Time manager
        time_manager = pp.TimeManager([0, 0.01, 0.1, 0.5, 1, 2], 0.001, constant_dt=True)

        # Model parameters
        params = {
                'alpha_biot': 1.0,  # [-]
                'height': 1.0,  # [m]
                'lambda_lame': 1.65E9,  # [Pa]
                'mu_lame': 1.475E9,  # [Pa]
                'num_cells': 20,
                'permeability': 9.86E-14,  # [m^2]
                'perturbation_factor': 1E-6,
                'plot_results': True,
                'specific_weight': 9.943E3,  # [Pa * m^-1]
                'time_manager': time_manager,
                'upper_limit_summation': 1000,
                'use_ad': True,
                'vertical_load': 6E8,  # [N * m^-1]
                'viscosity': 1E-3,  # [Pa * s]
            }

        # Run model
        tic = time()
        print("Simulation started...")
        model = Terzaghi(params)
        pp.run_time_dependent_model(model, params)
        print(f"Simulation finished in {round(time() - tic)} sec.")

    """

    def __init__(self, params: dict):
        """Constructor of the Terzaghi class.

        Args:
            params: Model parameters. Admissible entries are:

                - 'alpha_biot' : Biot constant (int or float).
                - 'height' : Height of the domain in `m` (int or float).
                - 'lambda_lame' : Lamé parameter in `Pa` (int or float).
                - 'mu_lame' : Lamé parameter in `m` (int or float).
                - 'num_cells' : Number of vertical cells (int).
                - 'permeability' : Intrinsic permeability in `m^2` (int or float).
                - 'pertubation_factor' : Perturbation factor (int or float). To be applied to
                  the vertical node coordinates of the mesh. This is necessary to avoid
                  singular matrices with MPSA and the use of roller boundary conditions.
                - 'plot_results' : Whether to plot the results (bool). The resulting plot is
                  saved inside the `out` folder.
                - 'specific_weight' : Fluid specific weight in `Pa * m^-1` (int or float).
                  Recall that the specific weight is density * gravity.
                - 'time_manager' : Time manager object (pp.TimeManager).
                - 'use_ad' : Whether to use ad (bool). Note that this must be set to True.
                  Otherwise, an error will be raised.
                - 'vertical_load' : Applied vertical load in `N * m^-1` (int or float).
                - 'viscosity' : Fluid viscosity in `Pa * s` (int or float).

        """
        super().__init__(params)

        # ad sanity check
        if not self.params["use_ad"]:
            raise ValueError("Model only valid when ad is used.")

        # Create a solution dictionary to store variables of interest
        self.sol = {counter: {} for counter in range(len(self.time_manager.schedule))}

        # Counter for storing variables of interest
        self._store_counter: int = 0

    def create_grid(self) -> None:
        """Create a two-dimensional Cartesian grid."""
        n = self.params["num_cells"]
        h = self.params["height"]
        phys_dims = np.array([h, h])
        n_cells = np.array([1, n])
        self.box = pp.geometry.bounding_box.from_points(
            np.array([[0, 0], phys_dims]).T
        )
        sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        sd.compute_geometry()
        np.random.seed(35)  # this seed is fixed but completely arbitrary
        # Perturb the y-coordinate of the physical nodes to avoid singular matrices with
        # roller bc and MPSA. For Terzaghi's problem, perturb only the vertical coordinates,
        # NOT the horizontal coordinates.
        perturbation_factor = self.params["perturbation_factor"]
        perturbation = np.random.rand(sd.num_nodes) * perturbation_factor
        sd.nodes[1] += perturbation
        sd.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])

    def _initial_condition(self) -> None:
        """Override initial condition for the flow subproblem."""
        super()._initial_condition()
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        vertical_load = self.params["vertical_load"]
        initial_p = vertical_load * np.ones(sd.num_cells)
        data[pp.STATE][self.scalar_variable] = initial_p
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = initial_p
        self._store_variables()

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            Scalar boundary condition representation.

        """
        # Define boundary regions
        all_bc, _, _, north, *_ = self._domain_boundary_sides(sd)
        north_bc = np.isin(all_bc, np.where(north)).nonzero()

        # All sides Neumann, except the North which is Dirichlet
        bc_type = np.asarray(all_bc.size * ["neu"])
        bc_type[north_bc] = "dir"

        bc = pp.BoundaryCondition(sd, faces=all_bc, cond=bc_type)

        return bc

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define boundary condition types for the mechanics subproblem

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Vectorial boundary condition representation.

        """
        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        super()._bc_type_mechanics(sd=sd)

        # Get boundary sides, retrieve data dict, and bc object
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Roller
        bc.is_neu[1, east] = True
        bc.is_dir[1, east] = False

        # West side: Roller
        bc.is_neu[1, west] = True
        bc.is_dir[1, west] = False

        # North side: Neumann
        bc.is_neu[:, north] = True
        bc.is_dir[:, north] = False

        # South side: Roller
        bc.is_neu[0, south] = True
        bc.is_dir[0, south] = False

        return bc

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            bc_values (sd.dim * sd.num_faces): Containing the boundary condition values.

        """

        # Retrieve boundary sides
        _, _, _, north, *_ = self._domain_boundary_sides(sd)

        # All zeros except vertical component of the north side
        vertical_load = self.params["vertical_load"]
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = -vertical_load * sd.face_areas[north]
        bc_values = bc_values.ravel("F")

        return bc_values

    def after_newton_convergence(
            self,
            solution: np.ndarray,
            errors: float,
            iteration_counter: int,
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Store solutions
        schedule = self.time_manager.schedule
        if any([np.isclose(self.time_manager.time, t_sch) for t_sch in schedule]):
            self._store_counter += 1  # increase exporter counter
            self._store_variables()  # store variables in the sol dictionary

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params["plot_results"]:
            self.plot_results()

    # -----> Physical parameters
    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Override value of intrinsic permeability [m^2]

        Args:
            sd: Subdomain grid.

        Returns:
            Array containing the permeability values on each cell.

        """
        return self.params["permeability"] * np.ones(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Override value of storativity [Pa^-1]

        Args:
            sd: Subdomain grid.

        Returns:
            Array containing the storativity values on each cell.

        """
        return np.zeros(sd.num_cells)

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Override stiffness tensor

        Args:
            sd: Subdomain grid.

        Returns:
            Fourth order tensorial representation of the stiffness tensor.

        """
        lam = (self.params["lambda_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        mu = (self.params["mu_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        return pp.FourthOrderTensor(mu, lam)

    def _viscosity(self, sd: pp.Grid) -> np.ndarray:
        """Override fluid viscosity values [Pa * s]

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
        """Confined compressibility [Pa^-1]"""

        mu_s = self.params["mu_lame"]
        lambda_s = self.params["lambda_lame"]
        m_v = 1 / (2 * mu_s + lambda_s)

        return m_v

    def consolidation_coefficient(self) -> Union[int, float]:
        """Consolidation coefficient [m^2 * s^-1]"""

        k = self.params["permeability"]  # [m^2]
        mu_f = self.params["viscosity"]  # [Pa * s]
        gamma_f = self.params["specific_weight"]  # [Pa * m^-1]
        hydraulic_conductivity = (k * gamma_f) / mu_f  # [m * s^-1]

        storativity = 0  # [Pa^-1]
        alpha_biot = self.params["alpha_biot"]  # [-]
        m_v = self.confined_compressibility()  # [Pa^-1]

        c_v = hydraulic_conductivity / (gamma_f * (storativity + alpha_biot ** 2 * m_v))

        return c_v

    # -----> Analytical expressions
    def nondim_time(self, t: Union[float, int]) -> float:
        """Nondimensionalize time.

        Args:
            t: Time in seconds.

        Returns:
            Dimensionless time for the given time `t`.

        """

        h = self.params["height"]
        c_v = self.consolidation_coefficient()

        return (t * c_v) / (h ** 2)

    def exact_pressure(self, t: Union[float, int]) -> np.ndarray:
        """Compute exact pressure.

        Args:
            t: Time in seconds.

        Returns:
            Exact pressure for the given time `t`.

        """

        sd = self.mdg.subdomains()[0]
        yc = sd.cell_centers[1]
        h = self.params["height"]
        vertical_load = self.params["vertical_load"]
        dimless_t = self.nondim_time(t)

        n = self.params["upper_limit_summation"]

        sum_series = np.zeros_like(yc)
        for i in range(1, n + 1):
            sum_series += (
                    (((-1) ** (i - 1)) / (2 * i - 1))
                    * np.cos((2 * i - 1) * (np.pi / 2) * (yc / h))
                    * np.exp((-((2 * i - 1) ** 2)) * (np.pi ** 2 / 4) * dimless_t)
            )
        p = (4 / np.pi) * vertical_load * sum_series

        return p

    # -----> Helper methods
    def _store_variables(self) -> None:
        """Utility function to store variables of interest."""

        # Useful data
        sd: pp.Grid = self.mdg.subdomains()[0]
        data: dict = self.mdg.subdomain_data(sd)
        t: Union[float, int] = self.time_manager.time
        p_var: str = self.scalar_variable
        u_var: str = self.displacement_variable
        p0: Union[float, int] = np.abs(self.params["vertical_load"])
        yc: np.ndarray = sd.cell_centers[1]
        height: Union[float, int] = self.params["height"]

        # Store solutions
        self.sol[self._store_counter]["t"] = t
        self.sol[self._store_counter]["t_nondim"] = self.nondim_time(t)
        self.sol[self._store_counter]["yc"] = yc
        self.sol[self._store_counter]["yc_nondim"] = yc / height

        if t == 0:
            self.sol[self._store_counter]["unum"] = np.zeros(sd.dim * sd.num_cells)
            self.sol[self._store_counter]["pnum"] = p0
            self.sol[self._store_counter]["pex"] = p0
            self.sol[self._store_counter]["pnum_nondim"] = np.ones(sd.num_cells)
            self.sol[self._store_counter]["pex_nondim"] = np.ones(sd.num_cells)
        else:
            self.sol[self._store_counter]["unum"] = data[pp.STATE][u_var]
            self.sol[self._store_counter]["pnum"] = data[pp.STATE][p_var]
            self.sol[self._store_counter]["pex"] = self.exact_pressure(t)
            self.sol[self._store_counter]["pnum_nondim"] = data[pp.STATE][p_var] / p0
            self.sol[self._store_counter]["pex_nondim"] = self.exact_pressure(t) / p0

        self.sol[self._store_counter]["p_error"] = self.l2_relative_error(
            sd,
            self.sol[self._store_counter]["pex"],
            self.sol[self._store_counter]["pnum"],
            is_cc=True,
            is_scalar=True,
        )

    def displacement_trace(
            self, displacement: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Project the displacement vector onto the faces.

        Args:
            displacement (sd.dim * sd.num_cells, ): displacement solution.
            pressure (sd.num_cells, ): pressure solution.

        Returns:
            trace_u (sd.dim * sd.num_faces, ): trace of the displacement.

        """

        # Rename arguments
        u = displacement
        p = pressure

        # Discretization matrices
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        bound_u_cell = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key][
            "bound_displacement_cell"
        ]
        bound_u_face = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key][
            "bound_displacement_face"
        ]
        bound_u_pressure = data[pp.DISCRETIZATION_MATRICES][
            self.mechanics_parameter_key
        ]["bound_displacement_pressure"]

        # Mechanical boundary values
        bc_vals = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc_values"]

        # Compute trace of the displacement
        trace_u = bound_u_cell * u + bound_u_face * bc_vals + bound_u_pressure * p

        return trace_u

    def plot_results(self):
        """Plot dimensionless pressure"""

        folder = "out/"
        fnamep = "pressure"
        extension = ".pdf"
        cmap = mcolors.ListedColormap(
            plt.cm.tab20.colors[: len(self.time_manager.schedule)]
        )

        # -----> Pressure plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for key in self.sol:
            ax.plot(
                self.sol[key]["pex_nondim"],
                self.sol[key]["yc_nondim"],
                color=cmap.colors[key],
            )
            ax.plot(
                self.sol[key]["pnum_nondim"],
                self.sol[key]["yc_nondim"],
                color=cmap.colors[key],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=cmap.colors[key],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${np.round(self.sol[key]['t_nondim'], 6)}",
            )
        ax.set_xlabel(r"$\tilde{p} = p/p_0$", fontsize=15)
        ax.set_ylabel(r"$\tilde{y} = y/h$", fontsize=15)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.set_title("Normalized pressure profiles", fontsize=16)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + fnamep + extension, bbox_inches="tight")
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
            l2_error: discrete L2-error of the quantity of interest.

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
                meas = sd.cell_faces.repat(sd.dim)

        numerator = np.sqrt(np.sum(meas * np.abs(true_val - approx_val) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_val) ** 2))
        l2_error = numerator / denominator

        return l2_error
