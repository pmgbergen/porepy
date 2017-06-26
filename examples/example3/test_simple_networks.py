"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import numpy as np

from porepy.fracs import meshing


def test_single_isolated_fracture(**kwargs):
    """
    A single fracture completely immersed in a boundary grid.
    """
    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.simplex_grid([f_1], domain, **kwargs)

    if kwargs.get('return_expected', False):
        expected = [1, 1, 0, 0]
        return grids, expected
    else:
        return grids


def test_two_intersecting_fractures(**kwargs):
    """
    Two fractures intersecting along a line.

    The example also sets different characteristic mesh lengths on the boundary
    and at the fractures.

    """

    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-.7, -.7, .8, .8]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}

    mesh_size = {'mode': 'constant', 'value': 0.5, 'bound_value': 1}
    kwargs['mesh_size'] = mesh_size

    grids = meshing.simplex_grid([f_1, f_2], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 2, 1, 0]
    else:
        return grids


def test_three_intersecting_fractures(**kwargs):
    """
    Three fractures intersecting, with intersecting intersections (point)
    """

    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-.7, -.7, .8, .8]])
    f_3 = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])

    # Add some parameters for grid size
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    mesh_size = {'mode': 'constant', 'value': 0.5, 'bound_value': 1}

    kwargs['mesh_size'] = mesh_size
    grids = meshing.simplex_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 6, 1]
    else:
        return grids


def test_one_fracture_intersected_by_two(**kwargs):
    """
    One fracture, intersected by two other (but no point intersections)
    """

    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-.7, -.7, .8, .8]])
    f_3 = f_2 + np.array([0.5, 0, 0]).reshape((-1, 1))

    # Add some parameters for grid size
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    mesh_size = {'mode': 'constant', 'value': 0.5, 'bound_value': 1}

    kwargs['mesh_size'] = mesh_size
    grids = meshing.simplex_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 2, 0]
    else:
        return grids


def test_split_into_octants(**kwargs):
    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [0, 0, 0, 0]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.simplex_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 6, 1]
    else:
        return grids
    return grids


def test_three_fractures_sharing_line_same_segment(**kwargs):
    """
    Three fractures that all share an intersection line. This can be considered
    as three intersection lines that coincide.
    """
    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.simplex_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 1, 0]
    else:
        return grids


def test_three_fractures_split_segments(**kwargs):
    """
    Three fractures that all intersect along the same line, but with the
    intersection between two of them forming an extension of the intersection
    of all three.
    """
    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-.5, -.5, .5, .5]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.simplex_grid([f_1, f_2, f_3], domain, **kwargs)
    if kwargs.get('return_expected', False):
        return grids, [1, 3, 3, 2]
    else:
        return grids


def test_two_fractures_L_intersection(**kwargs):
    """
    Two fractures sharing a segment in an L-intersection.
    """
    f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    f_2 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.simplex_grid([f_1, f_2], domain, **kwargs)
    if kwargs.get('return_expected', False):
        return grids, [1, 2, 1, 0]
    else:
        return grids


def test_two_fractures_L_intersection_part_of_segment(**kwargs):
    """
    Two fractures sharing what is a full segment for one, an part of a segment
    for the other.
    """
    f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.simplex_grid([f_1, f_2], domain, **kwargs)
    if kwargs.get('return_expected', False):
        return grids, [1, 2, 1, 0]
    else:
        return grids


def test_two_fractures_L_intersection_one_displaced(**kwargs):
    """
    Two fractures sharing what is a part of segments for both.
    """
    f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    f_2 = np.array([[0, 0, 0, 0], [0.5, 1.5, 1.5, 0.5], [0, 0, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.simplex_grid([f_2, f_1], domain, **kwargs)
    if kwargs.get('return_expected', False):
        return grids, [1, 2, 1, 0]
    else:
        return grids


