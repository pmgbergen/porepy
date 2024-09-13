"""The file tests the Domain class.

The test coverage is somewhat patchy for now, see however additional tests (primarily in
the bounding box functionality) in test_fracture_network.py.
"""
import numpy as np
import pytest

import porepy as pp


@pytest.mark.parametrize("bounding_box", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_init(bounding_box, dim):
    """Test the initialization of the Domain class, both from bounding box and from
    polytope. The test is performed in 2d and 3d, and for both the bounding box and
    polytope input options.

    Failure here would signify something is wrong with the translation between
    the two input options.
    """
    known_box = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    if dim == 3:
        known_box["zmin"] = 0
        known_box["zmax"] = 1

    if dim == 2:
        west = np.array([[0, 0], [0, 1]])
        east = np.array([[1, 1], [0, 1]])
        south = np.array([[0, 1], [0, 0]])
        north = np.array([[0, 1], [1, 1]])
        known_polytope = [west, east, south, north]
    else:
        west = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]).T
        east = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]).T
        south = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]).T
        north = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]]).T
        bottom = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        top = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]).T
        known_polytope = [west, east, south, north, bottom, top]

    if bounding_box:
        # Generate from bounding box
        domain = pp.Domain(bounding_box=known_box)
        # Generate bounding box without providing min values
        max_box = {key: val for key, val in known_box.items() if "max" in key}
        domain_default_min = pp.Domain(bounding_box=max_box)
        # Check that the two domains are the same
        assert domain == domain_default_min
        # Proceed with the rest of the tests for the domain with min values.

    else:
        # Generate from polytope
        domain = pp.Domain(polytope=known_polytope)

    # Check that the bounding box is correct
    # First check that the number of keys is the same
    assert len(domain.bounding_box) == len(known_box)
    # Then check that the values are the same
    for key, val in known_box.items():
        assert domain.bounding_box[key] == val

    # Check that the polytope is correct
    # First check that the number of faces is the same
    assert len(domain.polytope) == len(known_polytope)
    # Then check that the faces are the same. The order of the arrays may be different,
    # so for each face in the generated polytope, check that there is a matching face
    # (up to permutation) in the known polytope.
    for face in domain.polytope:
        # Check that the face is in the known polytope. To check for permutations, we
        # need to cycle through the columns of the known face arrays.
        assert any(
            np.allclose(face, np.roll(known_face, i, axis=1))
            for known_face in known_polytope
            for i in range(known_face.shape[1])
        )


@pytest.mark.parametrize("dim", [2, 3])
def test_domain_contains_point(dim):
    """Test the Domain.__contains__ method. Failure here would signify that the
    method is not working as intended.
    """
    if dim == 2:
        domain = pp.Domain(bounding_box={"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
        # Generate points inside the domain, on the boundary (both sides), and strictly
        # outside.
        points = [
            np.array([0.5, 0.5]),
            np.array([0, 0.5]),
            np.array([1, 0.5]),
            np.array([0.5, 0]),
            np.array([0.5, 1]),
            np.array([-0.5, 0.5]),
            np.array([1.5, 0.5]),
            np.array([0.5, -0.5]),
            np.array([0.5, 1.5]),
        ]
        is_in = [True, True, True, True, True, False, False, False, False]
    else:  # dimension is 3
        domain = pp.Domain(
            bounding_box={
                "xmin": 0,
                "xmax": 1,
                "ymin": 0,
                "ymax": 1,
                "zmin": 0,
                "zmax": 1,
            }
        )
        # Generate points inside the domain, on the boundary (both sides), and strictly
        # outside.
        points = [
            np.array([0.5, 0.5, 0.5]),
            np.array([0, 0.5, 0.5]),
            np.array([1, 0.5, 0.5]),
            np.array([0.5, 0, 0.5]),
            np.array([0.5, 1, 0.5]),
            np.array([0.5, 0.5, 0]),
            np.array([0.5, 0.5, 1]),
            np.array([-0.5, 0.5, 0.5]),
            np.array([1.5, 0.5, 0.5]),
            np.array([0.5, -0.5, 0.5]),
            np.array([0.5, 1.5, 0.5]),
            np.array([0.5, 0.5, -0.5]),
            np.array([0.5, 0.5, 1.5]),
        ]
        is_in = [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    for p, i in zip(points, is_in):
        assert (p in domain) == i


@pytest.mark.parametrize("g,known", [
    # A 2d grid, cells have unit size.
    (pp.CartGrid([2, 2]),
        {"west": np.array([0, 3]),
        "east": np.array([2, 5]),
        "south": np.array([6, 7]),
        "north": np.array([10, 11]),
        "bottom": np.array([], dtype=int),
        "top": np.array([], dtype=int)        
        }
    ),
    # A 2d grid with 2x2 cells, the full domain has unit size.
    (pp.CartGrid([2, 2], physdims=[1, 1]),
        {"west": np.array([0, 3]),
        "east": np.array([2, 5]),
        "south": np.array([6, 7]),
        "north": np.array([10, 11]),
        "bottom": np.array([], dtype=int),
        "top": np.array([], dtype=int)
        }
    ),
    # A 3d grid.
    (pp.CartGrid([2, 2, 2]),
        {"west": np.array([0, 3, 6, 9]),
        "east": np.array([2, 5, 8, 11]),
        "south": np.array([12, 13, 18, 19]),
        "north": np.array([16, 17, 22, 23]),
        "bottom": np.array([24, 25, 26, 27]),
        "top": np.array([32, 33, 34, 35])
        }
    )
])
def test_domain_sides_from_grid(g: pp.Grid, known: dict):
    """Test the Domain.sides_from_grid method.

    Parameters:
        g: The grid to test.
        known: Dictionary with keys "west", "east", "south", "north", "bottom", "top",
            and values being the indices of the faces in the grid that are on the
            corresponding side of the domain.

    """
    g.compute_geometry()
    # Get the sides from the function to be tested, and check that they are correct.
    sides = pp.domain.domain_sides_from_grid(g)
    for key, val in known.items():
        assert np.all(np.where(getattr(sides, key))[0] == val)