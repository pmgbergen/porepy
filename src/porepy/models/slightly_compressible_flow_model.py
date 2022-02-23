"""This module contains a simple extension of the incompressible flow by
including a time derivative of the pressure and constant compressibility.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import porepy as pp

logger = logging.getLogger(__name__)


class SlightlyCompressibleFlow(pp.models.incompressible_flow_model.IncompressibleFlow):
    """ This class extends the Incompressible flow model by including a
    cummulative term expressed through pressure and a constant compressibility
    coefficient. For a full documenation refer to the parent class.

    The simulation starts at time t=0.

    Overwritten methods include:
        1. prepare_simulation:
        Starting time set to 0, end time and timestep size optional paramters
        2. _set_parameters: compressibility added

    New methods:
        1. _compressibility: constant compressibility per cell

    Attributes:
        end_time (float): Upper limit of considered time interval
        time_step (float): time step size
        time_index (int): number of time loops passed.
        time (float): simulation time
    """

    def __init__(
        self,
        end_time: Optional[float] = 10.,
        time_step: Optional[float] = 0.1,
        params: Optional[Dict] = None
    ) -> None:
        """ 
        Parameters:
            end_time (optional float): specifices end time for simulation.
                Simulation stops when first time value bigger than end_time is 
                reached.
            time_step (optional float ): Timestep size. Currently only
                uniform timestepping is supported.
        """
        super().__init__(params)

        # attributes
        # assure positivity and correct type and format of input
        self.end_time = abs(float(end_time))
        self.time_step = abs(float(time_step))
        self.time_index = 0
        self.time: float = float(0)

        
    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameters fields of the data dictionaries are updated for all
        subdomains and edges (of codimension 1).
        """
        super()._set_parameters()

        for g, d in self.gb:

            pp.initialize_data(
                g,
                d,
                self.parameter_key,
                {
                    "mass_weight": self._compressibility(g)
                }
            )

    def _compressibility(self, g: pp.Grid) -> np.ndarray:
        """Unitary compressibility.

        Units: Pa^(-1)
        """
        return np.ones(g.num_cells)
      

    def _assign_discretizations(self) -> None:
        """Define equations through discretizations.

        Uses the Mpfa discretization of the parent class for the elliptic part
        and same boundary conditions coupling on internal boundaries.

        Implements an Explicit Euler discretization in time.
        """

        super()._assign_discretizations()
        
        # Collection of subdomains
        subdomains: List[pp.Grid] = [g for g, _ in self.gb]
        
        # AD representation of the mass operator
        accumulation_term = pp.ad.MassMatrixAd(self.parameter_key, subdomains)
        
        # Access to pressure ad variable
        p = self._ad.pressure
        time_step_ad = pp.ad.Scalar(self.time_step, "time step")
        
        accumulation_term = accumulation_term.mass * (
            p - p.previous_timestep()
        )
        accumulation_term /= time_step_ad
        
        #  Adding accumulation term to incompressible flow equations 
        self._eq_manager.equations["subdomain_flow"] += accumulation_term
        
        