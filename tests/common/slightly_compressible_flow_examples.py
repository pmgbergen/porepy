"""
This module contains examples of slightly compressible flow setups.
"""
import porepy as pp
import numpy as np

from typing import Union


class NonLinearSCF(pp.SlightlyCompressibleFlow):
    """
    A model setup for mono-dimensional non-linear slightly compressible flow .

    Note:
        Let $\Omega \time (0, T)$ be the space-time cylinder of interest, then the governing
        equations read:

        \frac{\partial \phi(p)}{\partial t} + div(q) = f,     in \Omega \times (0, T),
        q = - \frac{K}{mu} grad(p),                           in \Omega \times (0, T).

        The non-linearity is introduced through the porosity [1] such that:

        \phi(p) = \phi_0 \exp[c_0 (p - p_0)],

        where $\phi_0$ is the porosity at the reference pressure $p_0$, and $c_0$ is the
        porous medium compressibility.

        [1] Barry, D. A., Lockington, D. A., Jeng, D. S., Parlange, J. Y., Li, L.,
        & Stagnitti, F. (2007). Analytical approximations for flow in compressible, saturated,
        one-dimensional porous media. Advances in water resources, 30(4), 927-936.

    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)

    def create_grid(self):
        phys_dims = np.array([1, 1])
        n: int = self.params.get("num_cells", 4)
        n_cells = np.array([n, n])
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        sd.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])

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

        phi0 = 0.2  # Reference porosity
        p0 = 1.0  # Reference pressure
        c0 = 1E-03  # Compressibility

        if isinstance(p, pp.ad.Ad_array):
            phi = phi0 * pp.ad.exp(c0 * (p - p0))
        elif isinstance(p, np.ndarray):
            phi = phi0 * np.exp(c0 * (p - p0))
        else:
            raise TypeError("Expected pressure to be Ad_array or np.ndarray.")

        return phi

    def before_newton_iteration(self) -> None:
        pass

    def after_newton_iteration(self, sol) -> None:
        pass

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
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        p = data[pp.STATE][self.variable]
        pp.plot_grid(sd, p, plot_2d=True)

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True


#%% Runner
time_manager = pp.TimeManager(schedule=[0, 10, 20], dt_init=1, constant_dt=True)
params = {"use_ad": True, "num_cells": 20, "time_manager": time_manager}
model = NonLinearSCF(params=params)
pp.run_time_dependent_model(model, params)

sd = model.mdg.subdomains()[0]
data = model.mdg.subdomain_data(sd)
p = data[pp.STATE][model.variable]
