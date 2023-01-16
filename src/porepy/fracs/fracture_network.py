"""
Module containing a wrapper method to generate fracture networks.
"""

import porepy as pp
import warnings

from porepy.fracs.fracture_network_2d import LineFracture
from porepy.fracs.fracture_network_3d import PlaneFracture
from typing import Literal, Union

Fracture = Union[LineFracture, PlaneFracture]
FractureNetwork = Union[pp.FractureNetwork2d, pp.FractureNetwork3d]


def create_fracture_network(
        fractures: list[Fracture],
        domain: dict[str, pp.number],
        dimension: Literal[2, 3],
        tol: float = 1e-8,
        run_checks: bool = False
) -> FractureNetwork:
    """
    Create a fracture network.

    Parameters:
        fractures: List of fracture objects. ``PlaneFracture`` for ``nd = 3``, and
            ``LineFracture`` for ``nd = 2``. 
        domain: Bounding box dictionary with keys ``xmin``, ``xmax``, ``ymin``,
            ``ymax``, ``zmin``, and ``zmax``.
        dimension: Ambient dimension. Either `2` or `3`.
        tol: Geometric tolerance used in computations.
        run_checks: Run consistency checks during the network processing. Can be
            considered a limited debug mode. Only valid for ``dimension=3``.

    Raises
        - TypeError: If not all items of ``fractures`` are of the same type.
        - ValueError: If the fracture type does not match the given dimension.
        - ValueError: If the dimension is not 2 or 3.
        - Warning: If run_checks=True when dimension != 3.

    Returns:
        Fracture network object according to the dimensionality of the problem.

    """
    # Check that all items are of the same type if the fracture list is non-empty.
    if len(fractures) > 0:
        if not (
            all(isinstance(frac, pp.FractureNetwork2d) for frac in fractures)
        ) or not (
            all(isinstance(frac, pp.FractureNetwork2d) for frac in fractures)
        ):
            raise TypeError("All fracture objects must be of the same type.")

    # Check that the fracture type matches the dimensionality of the problem if the
    # list of fractures is non-empty,
    if len(fractures) > 0:
        if (
                dimension == 2 and not isinstance(fractures[0], pp.FractureNetwork2d)
        ) or (
                dimension == 3 and not isinstance(fractures[0], pp.FractureNetwork3d)
        ):
            raise ValueError("Fracture type is not consistent with the dimension.")

    # Check correct dimension
    if dimension not in [2, 3]:
        raise ValueError("Expected dimension 2 or 3.")

    # Warn if run_checks = True is given when dimension != 3
    if run_checks and dimension != 3:
        warnings.warn("run_checks has no effect if dimension != 3.")

    # Create fracture network
    if dimension == 2:
        fracture_network = pp.FractureNetwork2d(fractures, domain, tol)
    else:
        fracture_network = pp.FractureNetwork3d(fractures, domain, tol, run_checks)

    return fracture_network

