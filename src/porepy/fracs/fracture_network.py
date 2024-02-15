"""This module contains a wrapper function to generate fracture networks in 2D and
3D, without having to deal with dimension-specific objects and methodology.

"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

# Custom typings
FractureList = Union[List[pp.LineFracture], List[pp.PlaneFracture]]
FractureNetwork = Union[FractureNetwork2d, FractureNetwork3d]


def create_fracture_network(
    fractures: Optional[FractureList] = None,
    domain: Optional[pp.Domain] = None,
    tol: float = 1e-8,
    run_checks: bool = False,
) -> FractureNetwork:
    """Create a fracture network in dimensions 2 or 3.

    Examples:

        .. code:: python3

            import porepy as pp
            import numpy as np

            # Set a domain
            domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

            # Create a list of fractures
            horizontal_frac = pp.LineFracture(np.array([[0.5, 0.5], [0.25, 0.75]]))
            vertical_frac = pp.LineFracture(np.array([[0.25, 0.75], [0.5, 0.5]]))
            fractures = [horizontal_frac, vertical_frac]

            # Construct the fracture network
            fracture_network = pp.create_fracture_network(fractures, domain)

    Parameters:
        fractures: ``default=None``

            List of fractures. Each item of the list should be either an instance of
            :class:`~porepy.fracs.plane_fracture.PlaneFracture` for 3D or an instance
            of :class:`~porepy.fracs.line_fracture.LineFracture` for 2D. Default is
            ``None``, meaning, no fractures. If an empty list is given, it will be
            treated as ``None``.
        domain: ``default=None``

            Domain specification.
        tol: ``default=1e-8``

            Geometric tolerance used in the computations.
        run_checks: ``default=False``

            Run consistency checks during the network processing. Can be considered a
            limited debug mode. Only used for three-dimensional fracture networks.

    Raises:
        ValueError:
            If ``domain = None`` **and** ``fractures = None`` (or an empty list).
        ValueError:
            If the dimensions from ``domain`` and ``fractures`` do not match.
        TypeError:
            If not all items of ``fractures`` are of the same type.
        Warning:
            If ``run_checks = True`` when dimension is different from 3.

    Returns:
        Fracture network object according to the dimensionality of the problem.

    """

    # Interpret empty `fractures` list as None
    fracs: Optional[FractureList]
    if isinstance(fractures, list) and len(fractures) == 0:
        fracs = None
    else:
        fracs = fractures

    # Both ``fracs`` and ``domain`` cannot be ``None`` at the same time
    if fracs is None and domain is None:
        raise ValueError("'fractures' and 'domain' cannot both be empty/None.")

    # Check that all items are of the same type if the fracture list is non-empty.
    if fracs is not None:
        is_homo_2d = all([isinstance(frac, pp.LineFracture) for frac in fracs])
        is_homo_3d = all([isinstance(frac, pp.PlaneFracture) for frac in fracs])
        if not (is_homo_2d or is_homo_3d):
            raise TypeError("All fracture objects must be of the same type.")

    # At this point, we can infer the dimension of the fracture network
    if fracs is None and domain is not None:  # CASE 1: domain given, no fractures
        dim = domain.dim
    elif fracs is not None and domain is not None:  # CASE2 2: both are given
        # Infer dimension form the list of fractures
        if isinstance(fracs[0], pp.LineFracture):
            dim_from_fracs = 2
        else:
            dim_from_fracs = 3
        # Infer the dimension from the dimension
        dim_from_domain = domain.dim
        # Make sure both dimension match
        if not dim_from_domain == dim_from_fracs:
            msg = "The dimensions inferred from 'fractures' and 'domain' do not match."
            raise ValueError(msg)
        dim = dim_from_domain
    else:  # CASE 3: fractures given, no domain
        assert fracs is not None and domain is None  # needed to please mypy
        if isinstance(fracs[0], pp.LineFracture):
            dim = 2
        else:
            dim = 3

    # Warn if run_checks = True is given when dimension is different from 3
    if run_checks and dim != 3:
        warnings.warn("'run_checks=True' has no effect if dimension != 3.")

    # Finally, create and return according to dimensionality mypy complains fervently
    # since fracs can be either 2d or 3d. Since we created a robust dimensionality
    # check above, it is safe to ignore the argument type
    if dim == 2:
        fracture_network_2d = FractureNetwork2d(
            fractures=fracs, domain=domain, tol=tol  # type: ignore[arg-type]
        )
        return fracture_network_2d
    else:
        fracture_network_3d = FractureNetwork3d(
            fractures=fracs,  # type: ignore[arg-type]
            domain=domain,
            tol=tol,
            run_checks=run_checks,
        )
        return fracture_network_3d
