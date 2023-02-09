"""
Module containing a wrapper method to generate fracture networks in 2d and 3d.
"""

import warnings

import porepy as pp
from porepy.fracs.line_fracture import LineFracture
from porepy.fracs.plane_fracture import PlaneFracture
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

from typing import Optional, Union

# Custom typings
Fracture = Union[LineFracture, PlaneFracture]
FractureNetwork = Union[FractureNetwork2d, FractureNetwork3d]


def create_fracture_network(
        fractures: Optional[list[Fracture]] = None,
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
        run_checks: bool = False
) -> FractureNetwork:
    """Create a fracture network.

    Examples:

        .. code: python3

            import porepy as pp

            # Set a domain
            domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

            # Create a list of fractures
            horizontal_frac = pp.LineFracture(np.array([[0.5, 0.5], [0.25, 0.75]]))
            vertical_frac = pp.LineFracture(np.array([[0.25, 0.75], [0.5, 0.5]]))
            fractures = [horizontal_frac, vertical_frac]

            # Construct the fracture network
            fracture_network = pp.create_fracture_network(fractures, domain)

    Raises:
        - ValueError: If ``domain = None`` and ``fractures = None`` (or an empty list).
        - TypeError: If not all items of ``fractures`` are of the same type.
        - Warning: If ``run_checks = True`` when dimension is different from 3.

    Parameters:
        fractures: List of fractures. Each item of the list should be either an
            instance of :class:`~porepy.fracs.plane_fracture.PlaneFracture` for 3d or an
            instance of :class:`~porepy.fracs.line_fracture.LineFracture` for 2d.
            Default is ``None``, meaning, no fractures. If an empty list is given, it
            will be treated as ``None``.
        domain: Domain specfication. An instance of
            :class:`~porepy.geometry.domain.Domain`.
        tol: Geometric tolerance used in the computations. Default is ``1e-8``.
        run_checks: Run consistency checks during the network processing. Can be
            considered a limited debug mode. Only used for three-dimensional fracture
            networks. Default is ``False``.

    Returns:
        Fracture network object according to the dimensionality of the problem.

    """

    # Interpret empty `fractures` list as None
    fracs: Optional[list[Fracture]]
    if isinstance(fractures, list) and len(fractures) == 0:
        fracs = None
    else:
        fracs = fractures

    # Both `fracs` and `domain` cannot be None at the same time
    if fracs is None and domain is None:
        raise ValueError("'fractures' and 'domain' cannot both be empty/None.")

    # Check that all items are of the same type if the fracture list is non-empty.
    if fracs is not None:
        if not (
            all(isinstance(frac, LineFracture) for frac in fracs)
        ) or not (
            all(isinstance(frac, PlaneFracture) for frac in fracs)
        ):
            raise TypeError("All fracture objects must be of the same type.")

    # At this point, we can infer the dimension of the fracture network
    if fracs is None and domain is not None:  # domain given, no fractures
        dim = domain.dim
    elif fracs is not None and domain is not None:  # both are given
        dim_from_domain = domain.dim
        dim_from_fracs = 2 if isinstance(fracs[0], LineFracture) else 3
        assert dim_from_domain == dim_from_fracs  # make sure both dimensions match
        dim = dim_from_domain
    else:  # fractures are given, no domain
        dim = 2 if isinstance(fracs[0], LineFracture) else 3

    # Warn if run_checks = True is given when dimension is different from 3
    if run_checks and dim != 3:
        warnings.warn("'run_checks=True' has no effect if dimension != 3.")

    # Now, we create the fracture network
    if dim == 2:
        fracture_network = pp.FractureNetwork2d(
            fractures=fracs,
            domain=domain,
            tol=tol
        )
    else:
        fracture_network = pp.FractureNetwork3d(
            fractures=fracs,
            domain=domain,
            tol=tol,
            run_checks=run_checks
        )

    return fracture_network

