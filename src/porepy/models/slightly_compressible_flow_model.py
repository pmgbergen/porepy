"""This module contains a simple extension of the incompressible flow by
including a time derivative of the pressure and constant compressibility.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class _AdVariables(pp.models.incompressible_flow_model._AdVariables):
    time_step: pp.ad.Scalar


class SlightlyCompressibleFlow(pp.models.incompressible_flow_model.IncompressibleFlow):
    """This class extends the Incompressible flow model by including a
    cumulative term expressed through pressure and a constant compressibility
    coefficient. For a full documentation refer to the parent class.

    The simulation starts at time t=0.

    Overridden methods include:
        1. _set_parameters: compressibility added
        2. _assign_discretizations: Upgrade incompressible flow equations
            to slightly compressible

    New methods:
        1. _compressibility: constant compressibility per cell

    Attributes:
        time_manager: Time-stepping control manager.

    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        """
        Parameters:
            params (dict): Dictionary of parameters used to control the solution procedure.
                Some frequently used entries are file and folder names for export,
                mesh sizes...
        """
        if params is None:
            params = {}
        super().__init__(params)

        # Time manager
        self.time_manager = params.get(
            "time_manager",
            pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
        )
        self._ad = _AdVariables()

    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameters fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).
        """
        super()._set_parameters()

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd, data, self.parameter_key, {"mass_weight": self._compressibility(sd)}
            )

    def _compressibility(self, g: pp.Grid) -> np.ndarray:
        """Unitary compressibility.

        Units: Pa^(-1)
        """
        return np.ones(g.num_cells)

    def _assign_equations(self) -> None:
        """Upgrade incompressible flow equations to slightly compressible
            by adding the accumulation term.

        Time derivative is approximated with Implicit Euler time stepping.
        """

        super()._assign_equations()

        # AD representation of the mass operator
        accumulation_term = pp.ad.MassMatrixAd(self.parameter_key, self._ad.subdomains)

        # Access to pressure ad variable
        p = self._ad.pressure
        self._ad.time_step = pp.ad.Scalar(self.time_manager.dt, "time step")

        accumulation_term = (
            accumulation_term.mass * (p - p.previous_timestep()) / self._ad.time_step
        )

        #  Adding accumulation term to incompressible flow equations
        self._eq_manager.equations["subdomain_flow"] += accumulation_term

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu([self.variable], time_dependent=True)

    def after_simulation(self):
        self.exporter.write_pvd()

    def after_newton_convergence(
            self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """ Method to be called after the non-linear solver has converged.

        Args:
            solution: solution to the non-linear problem.
            errors: L2-norm of the absolute error.
            iteration_counter: number of non-linear iterations.

        """

        if not self.time_manager.is_constant:
            if not self._is_non_linear_problem():
                msg = "Currently, time step cannot be adapted when the problem is linear."
                raise NotImplementedError(msg)
            else:
                self.time_manager.compute_time_step(iterations=iteration_counter)
                if hasattr(self, "_ad"):
                    self._ad.time_step._value = self.time_manager.dt
                else:
                    raise NotImplementedError("Adaptive time step only available for AD.")
        else:
            # Nothing to here since the time step is constant
            pass

    def _is_time_dependent(self):
        return True

