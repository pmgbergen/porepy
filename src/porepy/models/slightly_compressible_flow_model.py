"""This module contains a simple extension of the incompressible flow by
including a time derivative of the pressure and constant compressibility.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class _AdVariables(pp.models.incompressible_flow_model._AdVariables):
    time_step: pp.ad.Scalar


class SlightlyCompressibleFlow(pp.models.incompressible_flow_model.IncompressibleFlow):
    """This class extends the Incompressible flow model by including a
    cummulative term expressed through pressure and a constant compressibility
    coefficient. For a full documenation refer to the parent class.

    The simulation starts at time t=0.

    Overwritten methods include:
        1. _set_parameters: compressibility added
        2. _assign_discretizations: Upgrade incompressible flow equations to slightly compressible

    New methods:
        1. _compressibility: constant compressibility per cell

    Attributes:
        end_time (float): Upper limit of considered time interval
        time_step (float): time step size
        time (float): simulation time
        time_index (int): number of time steps passed
    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        """
        Parameters:
            params (dict): Dictionary of parameters used to control the solution procedure.
                Some frequently used entries are file and folder names for export,
                mesh sizes...
        """
        super().__init__(params)

        # attributes
        if isinstance(params, type(None)):
            self.end_time = self.params.get("end_time", float(1))
            self.time_step = self.params.get("time_step", float(1))
        else:
            self.end_time = float(1)
            self.time_step = float(1)

        self.time = float(0)
        self.time_index = 0
        self._ad = _AdVariables()

    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameters fields of the data dictionaries are updated for all
        subdomains and edges (of codimension 1).
        """
        super()._set_parameters()

        for g, d in self.gb:

            pp.initialize_data(
                g, d, self.parameter_key, {"mass_weight": self._compressibility(g)}
            )

    def _compressibility(self, g: pp.Grid) -> np.ndarray:
        """Unitary compressibility.

        Units: Pa^(-1)
        """
        return np.ones(g.num_cells)

    def _assign_discretizations(self) -> None:
        """Upgrade incompressible flow equations to slightly compressible by adding the accumulation term.
        Time derivative is approximated with Implicit Euler time stepping.
        """

        super()._assign_discretizations()

        # Collection of subdomains
        subdomains: List[pp.Grid] = [g for g, _ in self.gb]

        # AD representation of the mass operator
        accumulation_term = pp.ad.MassMatrixAd(self.parameter_key, subdomains)

        # Access to pressure ad variable
        p = self._ad.pressure
        self._ad.time_step = pp.ad.Scalar(self.time_step, "time step")

        accumulation_term = (
            accumulation_term.mass * (p - p.previous_timestep()) / self._ad.time_step
        )

        #  Adding accumulation term to incompressible flow equations
        self._eq_manager.equations["subdomain_flow"] += accumulation_term
