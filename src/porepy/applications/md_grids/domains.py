from __future__ import annotations

from typing import Literal

import numpy as np

import porepy as pp


def nd_cube_domain(dimension: Literal[2, 3], size=pp.number) -> pp.Domain:
    """Return a domain extending from 0 to `size` in all dimensions.

    Parameters:

        dimension: The dimension of the domain.
        size: The side length of the cube.

    Returns:
        Dimension-cube domain object.

    """

    assert dimension in np.arange(2, 4)
    box: dict[str, pp.number] = {"xmax": size}

    if dimension > 1:
        box.update({"ymax": size})

    if dimension > 2:
        box.update({"zmax": size})

    return pp.Domain(box)


def unit_cube_domain(dimension: Literal[2, 3]) -> pp.Domain:
    """Return a domain of unitary size extending from 0 to 1 in all dimensions.

    Parameters:
        dimension: The dimension of the domain.

    Returns:
        Unit dimension-cube domain object.

    """

    return nd_cube_domain(dimension, 1)
