"""
This module contains examples of slightly compressible flow setups.

Implemented classes:
    NonLinearSCF: Two-dimensional non-fractured slightly compressible flow with
        pressure-dependent porosity.

"""
import porepy as pp
import numpy as np
import sympy as sym

from typing import Callable, Union


class NonLinearSCF(pp.SlightlyCompressibleFlow):
    """
    A model setup for mono-dimensional non-linear slightly compressible flow .

    Note:
        Let $\Omega \time (0, T)$ be the space-time cylinder of interest, then the governing
        equations read:

        \frac{\partial \phi(p)}{\partial t} + div(q) = f,     in \Omega \times (0, T),
        q = - \frac{k}{mu} grad(p),                           in \Omega \times (0, T).

        The non-linearity is introduced through the porosity [1]. To be precise, we let

        \phi(p) = \phi_0 \exp[c_0 (p - p_0)],

        where $\phi_0$ is the porosity at the reference pressure $p_0$, and $c_0$ is the
        porous medium compressibility (also referred to as the specific storativity).

        [1] Barry, D. A., Lockington, D. A., Jeng, D. S., Parlange, J. Y., Li, L.,
        & Stagnitti, F. (2007). Analytical approximations for flow in compressible, saturated,
        one-dimensional porous media. Advances in water resources, 30(4), 927-936.

    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)

        self.out = {"iterations": [], "time_step": []}

    def create_grid(self):
        phys_dims = np.array([1, 1])
        n: int = self.params.get("num_cells", 4)
        n_cells = np.array([n, n])
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        sd.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])

    def _reference_porosity(self):
        return self.params.get("reference_porosity", 0.5)

    def _reference_pressure(self):
        return self.params.get("reference_pressure", 0.0)

    def _specific_storativity(self):
        return self.params.get("specific_storativity", 1E-1)

    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        return self.params.get("permeability", 1.0) * np.ones(sd.num_cells)


    def _porosity(
            self, p: Union[pp.ad.Ad_array, np.ndarray]
        ) -> Union[pp.ad.Ad_array, np.ndarray]:
        """
        Porosity as a function of pressure.

        Args:
            p: pressure

        Returns:
            porosity for the given pressure `p`.

        """

        phi0 = self._reference_porosity()
        p0 = self._reference_pressure()
        c0 = self._specific_storativity()

        if isinstance(p, pp.ad.Ad_array):
            phi = phi0 * pp.ad.exp(c0 * (p - p0))
        elif isinstance(p, np.ndarray):
            phi = phi0 * np.exp(c0 * (p - p0))
        else:
            raise TypeError("Expected pressure to be Ad_array or np.ndarray.")

        return phi

    def before_newton_loop(self) -> None:

        super().before_newton_loop()

        # Update source term
        sd: pp.Grid = self.mdg.subdomains()[0]
        data: dict = self.mdg.subdomain_data(sd)
        vols: np.ndarray = sd.cell_volumes
        cc: np.ndarray = sd.cell_centers
        f: Callable = self.exact_source()
        t: float = self.time_manager.time
        source = vols * f(cc[0], cc[1], t)
        data[pp.PARAMETERS][self.parameter_key]["source"] = source

    def before_newton_iteration(self) -> None:
        pass

    def after_newton_iteration(self, sol) -> None:

        # Distribute solution to iterate
        self.dof_manager.distribute_variable(
            values=sol,
            grids=self.mdg.subdomains(),
            variables=self.variable,
            additive=True,
            to_iterate=True
        )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:

        # Store number of iterations and time step for testing purposes
        self.out["iterations"].append(iteration_counter)
        self.out["time_step"].append(self.time_manager.dt)
        print(self.time_manager)
        print()

        super().after_newton_convergence(solution, errors, iteration_counter)

        # Distribute variables to state
        # TODO: We might want to implement this in the parent class already, since it is
        #  quite easily overlooked
        self.dof_manager.transfer_variable(from_iterate_to_state=True)

    def after_newton_failure(
            self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:

        super().after_newton_failure(solution, errors, iteration_counter)

        # Since solution will be recomputed, transfer from state to iterate
        self.dof_manager.transfer_variable(from_state_to_iterate=True)

    def _assign_equations(self) -> None:
        """Upgrade incompressible flow equations to non-linear slightly compressible
            by adding the accumulation term.

        Time derivative is approximated with Implicit Euler time stepping.
        """

        super()._assign_equations()

        # AD representation of the mass operator
        accumulation_term = pp.ad.MassMatrixAd(self.parameter_key, self._ad.subdomains)

        # Access to pressure ad variable
        p = self._ad.pressure
        self._ad.time_step = pp.ad.Scalar(self.time_manager.dt, "time step")

        # Retrieve porosity
        poro = pp.ad.Function(self._porosity, name="porosity", array_compatible=True)

        accumulation_term = (
                accumulation_term.mass * (poro(p) - poro(p.previous_timestep()))
        ) / self._ad.time_step

        #  Adding accumulation term to incompressible flow equations
        self._eq_manager.equations["subdomain_flow"] += accumulation_term

    def after_simulation(self):
        if self.params.get("plot_sol", False):
            sd = self.mdg.subdomains()[0]
            data = self.mdg.subdomain_data(sd)
            p = data[pp.STATE][self.variable]
            pp.plot_grid(sd, p, plot_2d=True)

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

    def exact_source(self) -> Callable:
        """Determine the exact source term for a given manufactured solution.

        We consider two types of manufactured solutions, i.e., parabolic and trigonometric.

        The specification is done through the model params via the "solution_type" key.
        Admissible values for the "solution_type" key are "parabolic" and "trigonometric".
        "parabolic" is set if the key does not exist.

        Returns:
            Exact source lambda function of `x`, `y`, and `t`.

        """

        # Declare symbolic variables
        x, y, t = sym.symbols("x y t")

        # Physical parameters for the model
        phi0 = self._reference_porosity()
        p0 = self._reference_pressure()
        c0 = self._specific_storativity()

        # Determine exact source
        solution_type = self.params.get("solution_type", "parabolic")
        if solution_type == "parabolic":
            p_sym = t * x * (1 - x) * y * (1 - y)
        elif solution_type == "trigonometric":
            p_sym = t * sym.sin(2 * sym.pi * x) * sym.cos(2 * sym.pi * y)
        else:
            raise ValueError("Solution type must be either 'parabolic' or 'trigonometric'.")

        q_sym = [-sym.diff(p_sym, x), -sym.diff(p_sym, y)]
        divq_sym = sym.diff(q_sym[0], x) + sym.diff(q_sym[1], y)
        phi_sym = phi0 * sym.exp(c0 * (p_sym - p0))
        dphidt_sym = sym.diff(phi_sym, t)
        f_sym = dphidt_sym + divq_sym
        f_fun = sym.lambdify((x, y, t), f_sym, "numpy")

        return f_fun


#%% Runner
time_manager = pp.TimeManager(
    schedule=[0, 0.2],
    dt_init=0.19,
    dt_min_max=(0.09, 0.19),
    iter_optimal_range=(1, 2),
    print_info=True,
    iter_max=2,
    recomp_factor=0.5,
    recomp_max=5
)
params = {
    "use_ad": True,
    "num_cells": 5,
    "time_manager": time_manager,
    "solution_type": "trigonometric",
    "plot_sol": True,
    "max_iterations": 2,
    "nl_convergence_tol": 1E-6,
}
model = NonLinearSCF(params=params)
pp.run_time_dependent_model(model, params)

sd = model.mdg.subdomains()[0]
data = model.mdg.subdomain_data(sd)
p = data[pp.STATE][model.variable]
