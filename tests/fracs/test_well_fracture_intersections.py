"""Tests for the geometry of well-fracture intersections.

This module targets the (not yet implemented) function:

    def intersect_well_fractures(
        wells: list[pp.Well],
        fractures: list[pp.Fracture],
    ) -> tuple[tuple[np.ndarray, int, list[int]], ...]

where each entry of the returned tuple is itself a tuple of:
    (coordinate, well_index, [fracture_indices])

with:
    coordinate : np.ndarray, shape (ambient_dim,)
        The spatial coordinate of the intersection point.
    well_index : int
        Index into the input ``wells`` list for the well involved.
    fracture_indices : list[int]
        Indices (into the input ``fractures`` list) of all fractures that
        share this intersection point.

Design notes
------------
The tests are purely geometric; no mesh is constructed.
Geometry fixtures (wells, fractures) are exposed as module-level helpers
  so that a future meshing test module can import and reuse them directly,
  avoiding duplication.

"""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp

# Helper - import the function under test.
# This import will fail until the function is implemented; all tests that depend
# on it are therefore skipped so that the test file can be collected cleanly
# even before the function is available.
try:
    from porepy.geometry.intersections import intersect_well_fractures

    _FUNCTION_AVAILABLE = True
except ImportError:
    _FUNCTION_AVAILABLE = False


# Decorator applied to every test that calls the target function.
requires_implementation = pytest.mark.skipif(
    not _FUNCTION_AVAILABLE,
    reason=(
        "The function 'intersect_well_fractures' is not yet implemented (issue #1635)."
    ),
)


# Tolerance used throughout for coordinate comparisons.
TOL = 1e-10


# 3D fractures factories (PlaneFracture for ambient dim = 3)
def make_fracture_horizontal_at_z(
    fracture_index: int, z: float, half_size: float = 2.0
) -> pp.PlaneFracture:
    """Return a horizontal square fracture centered at (0,0,z).

    The fracture lies in the plane z = constant, and has vertices at
    (±half_size, ±half_size, z).

    Parameters:
        fracture_index: int
            Index to assign to the created fracture.
        z: float
            The z-coordinate of the fracture plane.
        half_size: float, optional
            Half the length of the sides of the square fracture.
    """

    pts = np.array(
        [
            [-half_size, half_size, half_size, -half_size],
            [-half_size, -half_size, half_size, half_size],
            [z, z, z, z],
        ]
    )
    return pp.PlaneFracture(pts, index=fracture_index)


def make_fracture_vertical_at_xz(
    fracture_index: int, y: float = 0.0, half_size: float = 2.0
) -> pp.PlaneFracture:
    """Return a vertical square fracture in the xz-plane at y=constant.

    The fracture lies in the plane y=y, span x in [-half_size, half_size],
    z in [-half_size, half_size].

    Parameters:
        fracture_index: int
            Index to assign to the created fracture.
        y: float
            The y-coordinate of the fracture plane.
        half_size: float, optional
            Half the length of the sides of the square fracture.

    """

    pts = np.array(
        [
            [-half_size, half_size, half_size, -half_size],
            [y, y, y, y],
            [-half_size, -half_size, half_size, half_size],
        ]
    )
    return pp.PlaneFracture(pts, index=fracture_index)


def make_fracture_vertical_at_yz(
    fracture_index: int, x: float = 0.0, half_size: float = 2.0
) -> pp.PlaneFracture:
    """Return a vertical square fracture in the yz-plane at x=constant.

    The fracture lies in the plane x= constant, span y in [-half_size, half_size],
    z in [-half_size, half_size].

    Parameters:
        fracture_index: int
            Index to assign to the created fracture.
        x: float
            The x-coordinate of the fracture plane.
        half_size: float, optional
            Half the length of the sides of the square fracture.

    """

    pts = np.array(
        [
            [x, x, x, x],
            [-half_size, half_size, half_size, -half_size],
            [-half_size, -half_size, half_size, half_size],
        ]
    )
    return pp.PlaneFracture(pts, index=fracture_index)


# 2D fractures factories (LineFracture for ambient dim = 2)
def make_fracture_horizontal_at_y(
    fracture_index: int, y: float, x_min: float = -2.0, x_max: float = 2.0
) -> pp.LineFracture:
    """Return a horizontal line fracture at height y in 2D.

    Parameters:
        fracture_index: int
            Index to assign to the created fracture.
        y: float
            The y-coordinate of the fracture line.
        x_min, x_max:
            Extent in x
    """

    pts = np.array([[x_min, x_max], [y, y]])
    return pp.LineFracture(points=pts, index=fracture_index)


def make_fracture_vertical_at_x(
    fracture_index: int, x: float, y_min: float = -2.0, y_max: float = 2.0
) -> pp.LineFracture:
    """Return a vertical line fracture at x = const in 2D.

    Parameters:
        fracture_index: int
            Index to assign to the created fracture.
        x: float
            The x-coordinate of the fracture line.
        y_min, y_max:
            Extent in y

    """

    pts = np.array([[x, x], [y_min, y_max]])
    return pp.LineFracture(points=pts, index=fracture_index)


# 3D well factories
def make_well_3d(
    well_index: int,
    points: list[tuple[float, float, float]],
) -> pp.Well:
    """Return a 3D well from ordered points (piecewise linear segments).

    Parameters:
        well_index: int
            Index to assign to the created well.
        points: list of tuples
            Each tuple is a (x, y, z) coordinate of a point along the well path, in order.
    """

    return pp.Well(np.array(points).T, well_index)


# 2D well factories
def make_well_2d(
    well_index: int,
    points: list[tuple[float, float]],
) -> pp.Well:
    """Return a 2D well from ordered points (piecewise linear segments).

    Parameters:
        well_index: int
            Index to assign to the created well.
        points: list of tuples
            Each tuple is a (x, y) coordinate of a point along the well path, in order.

    """

    return pp.Well(np.array(points).T, well_index)


# Result verification helpers
def _find_intersection(
    result: tuple,
    well_index: int,
    fracture_indices: set[int],
) -> np.ndarray | None:
    """Return the coordinate of the intersection matching well_index and fracture
    set, or None if not found.

    Parameters:
        result:
            Return value of `intersect_well_fractures`.
        well_index:
            Expected well index.
        fracture_indices:
            Expected set of fracture indices.

    """

    for coord, well_idx, frac_idxs in result:
        if well_idx == well_index and set(frac_idxs) == fracture_indices:
            return coord
    return None


@pytest.mark.parametrize(
    "case_name, well, fractures, expected",
    [
        (
            "2d_single_segment_middle",
            make_well_2d(
                0,
                [(0.0, 3.0), (0.0, -3.0)],
            ),
            [make_fracture_horizontal_at_y(0, 0.0, x_min=-1.0, x_max=1.0)],
            [(np.array([0.0, 0.0]), 0, [0])],
        ),
        (
            "2d_two_segment_kink",
            make_well_2d(
                0,
                [(0.0, 2.0), (0.0, 1.0), (1.0, -1.0)],
            ),
            [make_fracture_horizontal_at_y(0, 0.0, x_min=-1.0, x_max=1.0)],
            [(np.array([0.5, 0.0]), 0, [0])],
        ),
        (
            "2d_two_segment_bottom",
            make_well_2d(
                0,
                [(0.0, 2.0), (0.0, 1.0), (1.0, -1.0)],
            ),
            [make_fracture_horizontal_at_y(0, -1.0, x_min=-1.0, x_max=1.0)],
            [(np.array([1.0, -1.0]), 0, [0])],
        ),
        (
            "2d_multi_segment_multiple_intersections",
            make_well_2d(
                0,
                [(0.0, 2.0), (0.0, 1.0), (1.0, -1.0)],
            ),
            [
                make_fracture_horizontal_at_y(0, 1.0, x_min=-1.0, x_max=1.0),
                make_fracture_horizontal_at_y(1, 0.0, x_min=-1.0, x_max=1.0),
            ],
            [
                (np.array([0.0, 1.0]), 0, [0]),
                (np.array([0.5, 0.0]), 0, [1]),
            ],
        ),
        (
            "3d_single_segment_middle",
            make_well_3d(
                0,
                [(0.0, 0.0, 3.0), (0.0, 0.0, -3.0)],
            ),
            [make_fracture_horizontal_at_z(0, 1.0)],
            [(np.array([0.0, 0.0, 1.0]), 0, [0])],
        ),
        (
            "3d_two_segment_kink",
            make_well_3d(
                0,
                [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (1.0, 0.0, -3.0)],
            ),
            [make_fracture_horizontal_at_z(0, 0.0)],
            [(np.array([0.0, 0.0, 0.0]), 0, [0])],
        ),
        (
            "3d_two_segment_bottom",
            make_well_3d(
                0,
                [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (1.0, 0.0, -3.0)],
            ),
            [make_fracture_horizontal_at_z(0, -3.0)],
            [(np.array([1.0, 0.0, -3.0]), 0, [0])],
        ),
        (
            "3d_multi_segment_multiple_intersections",
            make_well_3d(
                0,
                [
                    (0.0, 0.0, 3.0),
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, -1.5),
                    (2.0, 0.0, -3.0),
                ],
            ),
            [
                make_fracture_horizontal_at_z(0, -0.75),
                make_fracture_horizontal_at_z(1, -2.25),
            ],
            [
                (np.array([0.5, 0.0, -0.75]), 0, [0]),
                (np.array([1.5, 0.0, -2.25]), 0, [1]),
            ],
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
@requires_implementation
def test_intersect_well_fractures_basic_geometries(
    case_name: str,
    well: pp.Well,
    fractures: list[pp.PlaneFracture | pp.LineFracture],
    expected: list[tuple[np.ndarray, int, list[int]]],
) -> None:
    """
    Test for `intersect_well_fractures` covering both 2D and 3D cases,
    as well as single- and multi-segment wells.

    The test verifies that:
        Intersection coordinates are computed correctly.
        The returned well index corresponds to the input.
        The correct fracture indices are associated with each intersection point.

    Covered scenarios:
        Ambient dimension:
            2D (line fractures)
            3D (planar fractures)

        Well geometry:
            Single-segment wells
            Multi-segment wells (with one or more kinks)

        Intersection locations along the well:
            Interior of a segment
            At a kink between two segments (must not produce duplicates)
            At the bottom (tip) of the well

        Multi-segment traversal:
            Intersections occurring on different segments of the same well
              are all detected and returned.

    Notes:
        The test is purely geometric; no mesh is constructed.
        Expected results are given as (coordinate, well_index, [fracture_indices]).
        Coordinates are compared using a numerical tolerance.

    """

    result = intersect_well_fractures([well], fractures)

    assert len(result) == len(expected), (
        f"Case '{case_name}': expected {len(expected)} intersections, "
        f"got {len(result)}."
    )

    for expected_coord, expected_well_idx, expected_frac_idxs in expected:
        coord = _find_intersection(
            result,
            well_index=expected_well_idx,
            fracture_indices=set(expected_frac_idxs),
        )
        assert coord is not None, (
            f"Case '{case_name}': missing intersection for well "
            f"{expected_well_idx} and fractures {expected_frac_idxs}."
        )
        np.testing.assert_allclose(coord, expected_coord, atol=TOL, err_msg=case_name)


@pytest.mark.parametrize(
    "case_name, well, fractures",
    [
        (
            "2d_no_intersections",
            make_well_2d(
                0,
                [(0.0, 3.5), (0.0, -3.5)],
            ),
            [
                make_fracture_horizontal_at_y(0, 4.0),  # above the well
                make_fracture_horizontal_at_y(1, -4.0),  # below the well
            ],
        ),
        (
            "3d_no_intersections",
            make_well_3d(
                0,
                [(0.0, 0.0, 3.5), (0.0, 0.0, -3.5)],
            ),
            [
                make_fracture_horizontal_at_z(0, 4.0),  # above the well
                make_fracture_horizontal_at_z(1, -4.0),  # below the well
            ],
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
@requires_implementation
def test_well_intersects_no_fractures(case_name, well, fractures) -> None:
    """Test that verify that a well with no geometric intersection returns an empty result."""

    result = intersect_well_fractures([well], fractures)

    assert len(result) == 0, (
        f"Case '{case_name}': expected no intersections, got {result}."
    )


@pytest.mark.parametrize(
    "case_name, wells, fractures, expected",
    [
        (
            "2d_single_fracture_multiple_wells",
            [
                make_well_2d(0, [(1.0, 2.0), (1.0, -2.0)]),
                make_well_2d(1, [(0.0, 2.0), (1.0, 1.0), (2.0, -2.0)]),
            ],
            [make_fracture_horizontal_at_y(0, 1.0, x_min=-1.0, x_max=3.0)],
            [
                (np.array([1.0, 1.0]), 0, [0]),
                (np.array([1.0, 1.0]), 1, [0]),
            ],
        ),
        (
            "3d_single_fracture_three_wells_mixed_segments",
            [
                make_well_3d(
                    0,
                    [(0.0, 0.0, 3.0), (0.0, 0.0, -3.0)],
                ),
                make_well_3d(
                    1,
                    [(1.0, 0.0, 3.0), (1.0, 0.0, 1.0), (2.0, 0.0, -2.0)],
                ),
                make_well_3d(
                    2,
                    [
                        (-1.0, 1.0, 3.0),
                        (-1.0, 1.0, 1.0),
                        (-0.5, 1.0, 0.0),
                        (0.5, 1.0, -2.0),
                    ],
                ),
            ],
            [make_fracture_horizontal_at_z(0, -1.0, half_size=3.0)],
            [
                (np.array([0.0, 0.0, -1.0]), 0, [0]),
                (np.array([1.5, 0.0, -1.0]), 1, [0]),
                (np.array([0.0, 1.0, -1.0]), 2, [0]),
            ],
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
@requires_implementation
def test_single_fracture_intersected_by_multiple_wells(
    case_name,
    wells,
    fractures,
    expected,
):
    """Test that verify that one fracture can be intersected by multiple wells
    and that one intersection record is returned per well-fracture intersection point.

    """

    result = intersect_well_fractures(wells, fractures)

    assert len(result) == len(expected), (
        f"Case '{case_name}': expected {len(expected)} intersections, "
        f"got {len(result)}."
    )

    for expected_coord, expected_well_idx, expected_frac_idxs in expected:
        coord = _find_intersection(
            result,
            well_index=expected_well_idx,
            fracture_indices=set(expected_frac_idxs),
        )
        assert coord is not None, (
            f"Case '{case_name}': missing intersection for well "
            f"{expected_well_idx} and fractures {expected_frac_idxs}."
        )
        np.testing.assert_allclose(coord, expected_coord, atol=TOL)


@pytest.mark.parametrize(
    "case_name, well, fractures, expected_coord, expected_fracture_indices",
    [
        (
            "codim_2_line_intersection",
            make_well_3d(
                0,
                [(0.0, 1.0, 1.0), (0.0, -1.0, -1.0)],
            ),
            [
                make_fracture_horizontal_at_z(0, 0.0, half_size=3.0),
                make_fracture_vertical_at_xz(1, y=0.0, half_size=3.0),
            ],
            np.array([0.0, 0.0, 0.0]),
            {0, 1},
        ),
        (
            "codim_2_line_intersection_shifted",
            make_well_3d(
                0,
                [(2.0, 3.0, 0.0), (2.0, -1.0, -2.0)],  # passes through y=1.0, z=-1.0
            ),
            [
                make_fracture_horizontal_at_z(0, -1.0, half_size=5.0),  # z = -1.0
                make_fracture_vertical_at_xz(1, y=1.0, half_size=5.0),  # y = 1.0
            ],
            np.array([2.0, 1.0, -1.0]),
            {0, 1},
        ),
        (
            "codim_3_point_intersection",
            make_well_3d(
                0,
                [(1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)],
            ),
            [
                make_fracture_horizontal_at_z(0, 0.0, half_size=3.0),
                make_fracture_vertical_at_xz(1, y=0.0, half_size=3.0),
                make_fracture_vertical_at_yz(2, x=0.0, half_size=3.0),
            ],
            np.array([0.0, 0.0, 0.0]),
            {0, 1, 2},
        ),
        (
            "codim_3_point_intersection_shifted",
            make_well_3d(
                0,
                [(2.5, 0.5, 3.0), (0.5, -1.5, 1.0)],  # passes through (1.5,-0.5,2.0)
            ),
            [
                make_fracture_horizontal_at_z(0, 2.0, half_size=5.0),  # z = 2.0
                make_fracture_vertical_at_xz(1, y=-0.5, half_size=5.0),  # y = -0.5
                make_fracture_vertical_at_yz(2, x=1.5, half_size=5.0),  # x = 1.5
            ],
            np.array([1.5, -0.5, 2.0]),
            {0, 1, 2},
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
@requires_implementation
def test_well_intersects_shared_fracture_intersection_in_3d(
    case_name,
    well,
    fractures,
    expected_coord,
    expected_fracture_indices,
) -> None:
    """
    Test verify that if a well intersects a shared fracture intersection set,
    the result contains a single intersection record with all fracture
    indices sharing that point.

    """

    result = intersect_well_fractures([well], fractures)

    assert len(result) == 1, (
        f"Case '{case_name}': expected exactly one same-point multi-fracture "
        f"intersection, got {result}."
    )

    coord = _find_intersection(
        result,
        well_index=0,
        fracture_indices=expected_fracture_indices,
    )
    assert coord is not None, (
        f"Case '{case_name}': expected one intersection shared by fractures "
        f"{sorted(expected_fracture_indices)}."
    )

    np.testing.assert_allclose(coord, expected_coord, atol=TOL)


@requires_implementation
def test_2d_multiple_fractures_same_point():
    """
    Test verify that a well intersecting the intersection point of two 2D fractures
    returns a single record with both fracture indices.

    """

    well = make_well_2d(0, [(1.0, 2.0), (1.0, -2.0)])

    fractures = [
        make_fracture_horizontal_at_y(0, 1.0),
        make_fracture_vertical_at_x(1, 1.0),
    ]

    result = intersect_well_fractures([well], fractures)

    assert len(result) == 1

    coord = _find_intersection(result, 0, {0, 1})
    assert coord is not None

    np.testing.assert_allclose(coord, np.array([1.0, 1.0]), atol=TOL)
