"""Utilities for functional tests."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DesiredValuesFlow:
    """Data class for storing desired errors and observed order of convergence."""

    error_matrix_pressure: float = -1.0
    error_matrix_flux: float = -1.0
    error_frac_pressure: float = -1.0
    error_frac_flux: float = -1.0
    error_intf_flux: float = -1.0
    ooc_matrix_pressure: float = -1.0
    ooc_matrix_flux: float = -1.0
    ooc_frac_pressure: float = -1.0
    ooc_frac_flux: float = -1.0
    ooc_intf_flux: float = -1.0

