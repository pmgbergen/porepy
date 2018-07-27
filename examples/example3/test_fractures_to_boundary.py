"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import numpy as np

from porepy.fracs import meshing


def kwarguments_and_domain():
    domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
    kwargs = {
        "mesh_size_frac": 1,
        "mesh_size_bound": 1,
        "mesh_size_min": .2,
        "return_expected": True,
    }

    return kwargs, domain


def check_number_of_grids(gb, expected):
    for d in range(4):
        assert len(gb.grids_of_dimension(3 - d)) == expected[d]


def test_one_fracture_touching_one_boundary(**kwargs):
    """
    A single fracture completely immersed in a boundary grid.
    """
    f_1 = np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])

    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        expected = [1, 1, 0, 0]
        return grids, expected
    else:
        return grids


def test_one_fracture_touching_two_neighbouring_boundaries(**kwargs):
    """
    Two fractures intersecting along a line.

    The example also sets different characteristic mesh lengths on the boundary
    and at the fractures.

    """

    f_1 = np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 2, 2]])

    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 1, 0, 0]
    else:
        return grids


def test_one_fracture_touching_two_opposing_boundaries(**kwargs):
    """
    Two fractures intersecting along a line.

    The example also sets different characteristic mesh lengths on the boundary
    and at the fractures.

    """

    f_1 = np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-1, -1, 1, 1]])

    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 1, 0, 0]

    else:
        return grids


def test_one_fracture_touching_three_boundaries(**kwargs):
    """
    Three fractures intersecting, with intersecting intersections (point)
    """

    f_1 = np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-1, -1, 2, 2]])
    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 1, 0, 0]
    else:
        return grids


def test_one_fracture_touches_four_boundaries(**kwargs):
    """
    One fracture, intersected by two other (but no point intersections)
    """

    f_1 = np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-2, -2, 2, 2]])

    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 1, 0, 0]
    else:
        return grids


def test_one_fracture_touches_boundary_corners(**kwargs):
    """
    One fracture, intersected by two other (but no point intersections)
    """

    f_1 = np.array([[-2, 2, 2, -2], [-2, 2, 2, -2], [-2, -2, 2, 2]])

    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 1, 0, 0]
    else:
        return grids


def test_two_fractures_touching_one_boundary(**kwargs):
    f_1 = np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, -1, 2, 2], [-1, 1, 1, -1], [0, 0, 0, 0]])
    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1, f_2], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 2, 1, 0]
    else:
        return grids
    return grids


def test_one_fracture_touches_boundary_corners(**kwargs):
    """
   One fracture running between opposite boundary corners
   """

    f_1 = np.array([[-2, 2, 2, -2], [-2, 2, 2, -2], [-2, -2, 2, 2]])

    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 2, 0, 0]
    else:
        return grids


def test_one_fracture_touches_boundary_and_corner(**kwargs):
    f_1 = np.array([[-2, 2, 2, -2], [-2, 1, 1, -2], [-2, -2, 2, 2]])

    kwargs, domain = kwarguments_and_domain()
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get("return_expected", False):
        return grids, [1, 2, 0, 0]
    else:
        return grids
