"""
Module for creating standard domain dictionaries.
"""
from typing import List, Dict


def CubeDomain(physdims: List[float]) -> Dict:
    """
    Create a domain dictionary of a Cube domain with lower left corner
    centered at Origin, and upper right corner centered at physdims

    Parameters:
    physdims (List) : Upper right corner of domain

    Returns:
    domain (Dict): domain dictionary.
    """
    for value in physdims:
        if value <= 0:
            msg = "physical dimension {} is not positive".format(value)
            raise ValueError(msg)

    domain = {"xmin": 0}
    domain["xmax"] = physdims[0]
    if len(physdims) > 3 or len(physdims) == 0:
        msg = "Invalid size of number of cells: {}".format(len(physdims))
        raise ValueError(msg)

    if len(physdims) >= 2:
        domain["ymin"] = 0
        domain["ymax"] = physdims[1]
    if len(physdims) >= 3:
        domain["zmin"] = 0
        domain["zmax"] = physdims[2]

    return domain


def SquareDomain(physdims: List[float]) -> Dict:
    """
    Create a domain dictionary of a Square domain with lower left corner
    centered at Origin, and upper right corner centered at physdims

    Parameters:
    physdims (List) : Upper right corner of domain

    Returns:
    domain (Dict): domain dictionary.
    """
    return CubeDomain(physdims)


def UnitCubeDomain() -> Dict:
    """
    Create a domain dictionary of a Unit cube

    Returns:
    {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 1, "ymax": 1, "zmax": 1}

    """
    return CubeDomain([1, 1, 1])


def UnitSquareDomain() -> Dict:
    """
    Create a domain dictionary of a Unit square

    Returns:
    {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

    """
    return CubeDomain([1, 1])
