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
from dataclasses import dataclass

import porepy as pp
from porepy.fracs.well_network import WellNetwork3d

# Tolerance used throughout for coordinate comparisons.
TOL = 1e-10


@pytest.fixture(autouse=True)
def finalize_gmsh():
    """Fixture to ensure gmsh is finalized after each test.

    This is to avoid tests failing because gmsh was not cleared after a previously
    breaking test.
    """
    yield  # This is where the test runs
    try:
        # Try to clear and finalize gmsh after each test. This will raise an error
        # if gmsh was not initialized in the test, but we can ignore that.
        gmsh.clear()
        gmsh.finalize()
    except Exception:
        pass


@dataclass(frozen=True)
class IntersectionCase:
    name: str
    wells: list[pp.Well]
    fractures: list[pp.PlaneFracture | pp.LineFracture]


def _infer_dimension_from_fractures(fractures: list[pp.Fracture]) -> int:
    if len(fractures) == 0:
        # Default to 3d if no fractures provided. The value given should not impact the
        # test results in this case.
        return 3
    return 2 if isinstance(fractures[0], pp.LineFracture) else 3


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
            Each tuple is a (x, y, z) coordinate of a point along the
            well path, in order.
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
    points = [(*pt, 0.0) for pt in points]  # Add z=0 for 2D wells to fit 3D interface

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

    for coord, _, well_idx, frac_idxs, _ in result:
        if well_idx == well_index and set(frac_idxs) == fracture_indices:
            return coord
    return None


def _run_case(case: IntersectionCase) -> tuple:
    well_network = WellNetwork3d(case.wells, None)
    fracture_network = pp.create_fracture_network(case.fractures, domain=None)

    return well_network.intersect_well_fractures(
        fracture_network.fractures, fracture_network.nd
    )[0]


def _assert_case_result(
    case: IntersectionCase,
    result: tuple,
    expected: list[tuple[np.ndarray, int, list[int]]],
) -> None:
    assert len(result) == len(expected), (
        f"Case '{case.name}': expected {len(expected)} intersections, "
        f"got {len(result)}."
    )

    empty_fracture_list = []

    for ind, (expected_coord, expected_well_idx, expected_frac_idxs) in enumerate(
        expected
    ):
        if len(expected_frac_idxs) == 0:
            # Special treatment of multiwell kinks.
            empty_fracture_list.append(ind)
            continue
        coord = _find_intersection(
            result,
            well_index=expected_well_idx,
            fracture_indices=set(expected_frac_idxs),
        )
        assert coord is not None, (
            f"Case '{case.name}': missing intersection for well "
            f"{expected_well_idx} and fractures {expected_frac_idxs}."
        )
        np.testing.assert_allclose(coord, expected_coord, atol=TOL, err_msg=case.name)

    covered = []
    for ind in empty_fracture_list:
        coord = expected[ind][0]
        for res_coord, _, well_idx, frac_idxs, _ in result:
            if set(frac_idxs) == set() and np.allclose(res_coord, coord, atol=TOL):
                assert well_idx == expected[ind][1], (
                    f"Case '{case.name}': unexpected well index for intersection at {coord}."
                )
                covered.append(ind)
                break
    assert len(covered) == len(empty_fracture_list), (
        f"Case '{case.name}': expected {len(empty_fracture_list)} intersections with no fractures, "
        f"got {len(covered)}."
    )


BASIC_GEOMETRY_CASES = (
    IntersectionCase(
        "2d_single_segment_middle",
        [make_well_2d(0, [(0.0, 3.0), (0.0, -3.0)])],
        [make_fracture_horizontal_at_y(0, 0.0, x_min=-1.0, x_max=1.0)],
    ),
    IntersectionCase(
        "2d_two_segment_kink",
        [make_well_2d(0, [(0.0, 2.0), (0.0, 1.0), (1.0, -1.0)])],
        [make_fracture_horizontal_at_y(0, 0.0, x_min=-1.0, x_max=1.0)],
    ),
    IntersectionCase(
        "2d_two_segment_bottom",
        [make_well_2d(0, [(0.0, 2.0), (0.0, 1.0), (1.0, -1.0)])],
        [make_fracture_horizontal_at_y(0, -1.0, x_min=-1.0, x_max=1.0)],
    ),
    IntersectionCase(
        "2d_multi_segment_multiple_intersections",
        [make_well_2d(0, [(0.0, 2.0), (0.0, 1.0), (1.0, -1.0)])],
        [
            make_fracture_horizontal_at_y(0, 1.0, x_min=-1.0, x_max=1.0),
            make_fracture_horizontal_at_y(1, 0.0, x_min=-1.0, x_max=1.0),
        ],
    ),
    IntersectionCase(
        "3d_single_segment_middle",
        [make_well_3d(0, [(0.0, 0.0, 3.0), (0.0, 0.0, -3.0)])],
        [make_fracture_horizontal_at_z(0, 1.0)],
    ),
    IntersectionCase(
        "3d_two_segment_kink",
        [make_well_3d(0, [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (1.0, 0.0, -3.0)])],
        [make_fracture_horizontal_at_z(0, 0.0)],
    ),
    IntersectionCase(
        "3d_two_segment_bottom",
        [make_well_3d(0, [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (1.0, 0.0, -3.0)])],
        [make_fracture_horizontal_at_z(0, -3.0)],
    ),
    IntersectionCase(
        "3d_multi_segment_multiple_intersections",
        [
            make_well_3d(
                0,
                [
                    (0.0, 0.0, 3.0),
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, -1.5),
                    (2.0, 0.0, -3.0),
                ],
            )
        ],
        [
            make_fracture_horizontal_at_z(0, -0.75),
            make_fracture_horizontal_at_z(1, -2.25),
        ],
    ),
)

BASIC_GEOMETRY_EXPECTED = {
    "2d_single_segment_middle": [(np.array([0.0, 0.0, 0.0]), 0, [0])],
    "2d_two_segment_kink": [
        (np.array([0.5, 0.0, 0.0]), 0, [0]),
        (np.array([0.0, 1.0, 0.0]), 0, []),
    ],
    "2d_two_segment_bottom": [
        (np.array([0.0, 1.0, 0.0]), 0, []),
        (np.array([1.0, -1.0, 0.0]), 0, [0]),
    ],
    "2d_multi_segment_multiple_intersections": [
        (np.array([0.0, 1.0, 0.0]), 0, [0]),
        (np.array([0.5, 0.0, 0.0]), 0, [1]),
    ],
    "3d_single_segment_middle": [(np.array([0.0, 0.0, 1.0]), 0, [0])],
    "3d_two_segment_kink": [(np.array([0.0, 0.0, 0.0]), 0, [0])],
    "3d_two_segment_bottom": [
        (np.array([0.0, 0.0, 0.0]), 0, []),
        (np.array([1.0, 0.0, -3.0]), 0, [0]),
    ],
    "3d_multi_segment_multiple_intersections": [
        (np.array([0.5, 0.0, -0.75]), 0, [0]),
        (np.array([1.5, 0.0, -2.25]), 0, [1]),
        (np.array([0.0, 0.0, 0.0]), 0, []),
        (np.array([1.0, 0.0, -1.5]), 0, []),
    ],
}

NO_INTERSECTION_CASES = (
    IntersectionCase(
        "2d_no_intersections",
        [make_well_2d(0, [(0.0, 3.5), (0.0, -3.5)])],
        [
            make_fracture_horizontal_at_y(0, 4.0),
            make_fracture_horizontal_at_y(1, -4.0),
        ],
    ),
    IntersectionCase(
        "3d_no_intersections",
        [make_well_3d(0, [(0.0, 0.0, 3.5), (0.0, 0.0, -3.5)])],
        [
            make_fracture_horizontal_at_z(0, 4.0),
            make_fracture_horizontal_at_z(1, -4.0),
        ],
    ),
)

MULTI_WELL_CASES = (
    IntersectionCase(
        "2d_single_fracture_multiple_wells",
        [
            make_well_2d(0, [(2.0, 2.0), (2.0, -2.0)]),
            make_well_2d(1, [(0.0, 2.0), (1.0, 1.0), (2.0, -2.0)]),
        ],
        [make_fracture_horizontal_at_y(0, 1.0, x_min=-1.0, x_max=3.0)],
    ),
    IntersectionCase(
        "3d_single_fracture_three_wells_mixed_segments",
        [
            make_well_3d(0, [(0.0, 0.0, 3.0), (0.0, 0.0, -3.0)]),
            make_well_3d(1, [(1.0, 0.0, 3.0), (1.0, 0.0, 1.0), (2.0, 0.0, -2.0)]),
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
    ),
)

MULTI_WELL_EXPECTED = {
    "2d_single_fracture_multiple_wells": [
        (np.array([2.0, 1.0, 0.0]), 0, [0]),
        (np.array([1.0, 1.0, 0.0]), 1, [0]),
    ],
    "3d_single_fracture_three_wells_mixed_segments": [
        (np.array([0.0, 0.0, -1.0]), 0, [0]),
        (np.array([5 / 3, 0.0, -1.0]), 1, [0]),
        (np.array([0.0, 1.0, -1.0]), 2, [0]),
        (np.array([1.0, 0.0, 1.0]), 1, []),
        (np.array([-1.0, 1.0, 1.0]), 2, []),
        (np.array([-0.5, 1.0, 0.0]), 2, []),
    ],
}

SHARED_INTERSECTION_CASES = (
    IntersectionCase(
        "codim_2_line_intersection",
        [make_well_3d(0, [(0.0, 1.0, 1.0), (0.0, -1.0, -1.0)])],
        [
            make_fracture_horizontal_at_z(0, 0.0, half_size=3.0),
            make_fracture_vertical_at_xz(1, y=0.0, half_size=3.0),
        ],
    ),
    IntersectionCase(
        "codim_2_line_intersection_shifted",
        [make_well_3d(0, [(2.0, 3.0, 0.0), (2.0, -1.0, -2.0)])],
        [
            make_fracture_horizontal_at_z(0, -1.0, half_size=5.0),
            make_fracture_vertical_at_xz(1, y=1.0, half_size=5.0),
        ],
    ),
    IntersectionCase(
        "codim_3_point_intersection",
        [make_well_3d(0, [(1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)])],
        [
            make_fracture_horizontal_at_z(0, 0.0, half_size=3.0),
            make_fracture_vertical_at_xz(1, y=0.0, half_size=3.0),
            make_fracture_vertical_at_yz(2, x=0.0, half_size=3.0),
        ],
    ),
    IntersectionCase(
        "codim_3_point_intersection_shifted",
        [make_well_3d(0, [(2.5, 0.5, 3.0), (0.5, -1.5, 1.0)])],
        [
            make_fracture_horizontal_at_z(0, 2.0, half_size=5.0),
            make_fracture_vertical_at_xz(1, y=-0.5, half_size=5.0),
            make_fracture_vertical_at_yz(2, x=1.5, half_size=5.0),
        ],
    ),
)

SHARED_INTERSECTION_EXPECTED = {
    "codim_2_line_intersection": (np.array([0.0, 0.0, 0.0]), {0, 1}),
    "codim_2_line_intersection_shifted": (np.array([2.0, 1.0, -1.0]), {0, 1}),
    "codim_3_point_intersection": (np.array([0.0, 0.0, 0.0]), {0, 1, 2}),
    "codim_3_point_intersection_shifted": (np.array([1.5, -0.5, 2.0]), {0, 1, 2}),
}


MULTIPLE_FRACTURES_SAME_POINT_2D_CASES = IntersectionCase(
    "multiple_fractures_same_point_2d",
    [make_well_2d(0, [(0.0, 2.0), (2.0, 0.0)])],
    [
        make_fracture_horizontal_at_y(0, 1.0),
        make_fracture_vertical_at_x(1, 1.0),
    ],
)

MULTIPLE_FRACTURES_SAME_POINT_2D_EXPECTED = {
    "multiple_fractures_same_point_2d": (np.array([1.0, 1.0, 0.0]), {0, 1}),
}

NO_INTERSECTION_CASE_NAMES = {case.name for case in NO_INTERSECTION_CASES}


def _expected_meshing_intersections(
    case: IntersectionCase,
) -> list[tuple[np.ndarray, int, list[int]]]:
    if case.name in BASIC_GEOMETRY_EXPECTED:
        return BASIC_GEOMETRY_EXPECTED[case.name]
    if case.name in MULTI_WELL_EXPECTED:
        return MULTI_WELL_EXPECTED[case.name]
    if case.name in SHARED_INTERSECTION_EXPECTED:
        coord, fracture_indices = SHARED_INTERSECTION_EXPECTED[case.name]
        return [(coord, 0, sorted(fracture_indices))]
    if case.name in MULTIPLE_FRACTURES_SAME_POINT_2D_EXPECTED:
        coord, fracture_indices = MULTIPLE_FRACTURES_SAME_POINT_2D_EXPECTED[case.name]
        return [(coord, 0, sorted(fracture_indices))]
    if case.name in NO_INTERSECTION_CASE_NAMES:
        return []
    raise ValueError(f"No meshing expectations configured for case '{case.name}'.")


def _assert_intersection_meshing(
    case: IntersectionCase,
    expected: list[tuple[np.ndarray, int, list[int]]],
) -> None:
    import gmsh

    nd = _infer_dimension_from_fractures(case.fractures)

    box = {
        "xmin": -5.0,
        "xmax": 5.0,
        "ymin": -5.0,
        "ymax": 5.0,
    }
    if nd == 3:
        box["zmin"] = -5.0
        box["zmax"] = 5.0

    domain = pp.Domain(box)
    fracture_network = pp.create_fracture_network(case.fractures, domain=domain)
    well_network = pp.WellNetwork3d(case.wells, domain)

    tmp_mdg = pp.create_mdg(
        "simplex",
        {
            "cell_size": 5.0,
            "refinement_proximity_multiplier": 1e-6,
            # Set values for mesh coarsening that will have minimal impact on the mesh.
            "refinement_size_multiplier": 1.0,
            "background_transition_multiplier": 1.01,
        },
        fracture_network=fracture_network,
    )

    num_frac_subdomains = len(tmp_mdg.subdomains(dim=nd - 1))
    num_fracture_intersections = len(tmp_mdg.subdomains(dim=nd - 2))
    if nd == 3:
        num_fracture_intersection_points = len(tmp_mdg.subdomains(dim=0))
    else:
        num_fracture_intersection_points = 0

    mdg = well_network.mesh(fracture_network, tmp_mdg, {"cell_size": 5.0})
    gmsh.clear()
    gmsh.finalize()
    num_intersections = len(expected)

    assert len(mdg.subdomains(dim=nd)) == 1
    if nd == 3:
        assert len(mdg.subdomains(dim=2)) == len(case.fractures)
        assert (
            len(mdg.subdomains(dim=1)) == len(case.wells) + num_fracture_intersections
        )
        assert (
            len(mdg.subdomains(dim=0))
            == num_intersections + num_fracture_intersection_points
        )
    if nd == 2:
        assert len(mdg.subdomains(dim=1)) == len(case.fractures) + len(case.wells)
        assert (
            len(mdg.subdomains(dim=0)) == num_intersections + num_fracture_intersections
        )

    found_intersection = num_intersections * [False]

    for pg in mdg.subdomains(dim=0):
        for i, (expected_coord, expected_well_idx, expected_frac_idxs) in enumerate(
            expected
        ):
            if np.allclose(pg.cell_centers[:, 0], expected_coord, atol=TOL):
                neigh_subdomains = mdg.neighboring_subdomains(pg, only_higher=True)
                if len(expected_frac_idxs) == 0:
                    assert len(neigh_subdomains) == 1
                else:
                    assert len(neigh_subdomains) >= 2
                num_fractures_found = 0
                for sd in neigh_subdomains:
                    if sd.well_num > -1:
                        assert sd.well_num == expected_well_idx, (
                            f"Case '{case.name}': expected well index  "
                            f"{expected_well_idx}, got {sd.well_num}."
                        )
                    elif sd.frac_num > -1:
                        # This is a fracture (and not a fracture intersection)
                        # subdomain.
                        assert sd.frac_num in expected_frac_idxs
                        num_fractures_found += 1
                    else:
                        # This is a fracture intersection
                        assert len(expected_frac_idxs) > 1, (
                            f"Case '{case.name}': expected multiple fractures at "
                            f"intersection, got {expected_frac_idxs}."
                        )
                        assert num_fractures_found == 0, (
                            f"Case '{case.name}': A well cannot intersect both a "
                            "fracture and a fracture intersection at the same point."
                        )

                found_intersection[i] = True
                break
    assert all(found_intersection), (
        f"Case '{case.name}': not all expected intersections were found in the mesh."
    )


@pytest.mark.parametrize("case", BASIC_GEOMETRY_CASES, ids=lambda case: case.name)
def test_intersect_well_fractures_basic_geometries(case: IntersectionCase) -> None:
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
    result = _run_case(case)
    _assert_case_result(case, result, BASIC_GEOMETRY_EXPECTED[case.name])


@pytest.mark.parametrize("case", NO_INTERSECTION_CASES, ids=lambda case: case.name)
def test_well_intersects_no_fractures(case: IntersectionCase) -> None:
    """Test that verify that a well with no geometric intersection returns
    an empty result.

    """

    result = _run_case(case)

    assert len(result) == 0, (
        f"Case '{case.name}': expected no intersections, got {result}."
    )


@pytest.mark.parametrize("case", MULTI_WELL_CASES, ids=lambda case: case.name)
def test_single_fracture_intersected_by_multiple_wells(case: IntersectionCase) -> None:
    """Test that verify that one fracture can be intersected by multiple wells
    and that one intersection record is returned per well-fracture intersection point.

    """

    result = _run_case(case)
    _assert_case_result(case, result, MULTI_WELL_EXPECTED[case.name])


@pytest.mark.parametrize("case", SHARED_INTERSECTION_CASES, ids=lambda case: case.name)
def test_well_intersects_shared_fracture_intersection_in_3d(
    case: IntersectionCase,
) -> None:
    """
    Test verify that if a well intersects a shared fracture intersection set,
    the result contains a single intersection record with all fracture
    indices sharing that point.

    """

    result = _run_case(case)

    assert len(result) == 1, (
        f"Case '{case.name}': expected exactly one same-point multi-fracture "
        f"intersection, got {result}."
    )

    expected_coord, expected_fracture_indices = SHARED_INTERSECTION_EXPECTED[case.name]
    coord = _find_intersection(
        result,
        well_index=0,
        fracture_indices=expected_fracture_indices,
    )
    assert coord is not None, (
        f"Case '{case.name}': expected one intersection shared by fractures "
        f"{sorted(expected_fracture_indices)}."
    )

    np.testing.assert_allclose(coord, expected_coord, atol=TOL)


def test_2d_multiple_fractures_same_point():
    """
    Test verify that a well intersecting the intersection point of two 2D fractures
    returns a single record with both fracture indices.

    """
    case = MULTIPLE_FRACTURES_SAME_POINT_2D_CASES
    result = _run_case(case)

    assert len(result) == 1
    expected_coord, expected_fracture_indices = (
        MULTIPLE_FRACTURES_SAME_POINT_2D_EXPECTED[case.name]
    )
    coord = _find_intersection(result, 0, expected_fracture_indices)
    assert coord is not None
    np.testing.assert_allclose(coord, expected_coord, atol=TOL)


@pytest.mark.parametrize("case", BASIC_GEOMETRY_CASES, ids=lambda case: case.name)
def test_basic_geometry_meshing(case: IntersectionCase) -> None:
    """Test basic geometry meshing for well-fracture intersections."""

    _assert_intersection_meshing(case, BASIC_GEOMETRY_EXPECTED[case.name])


@pytest.mark.parametrize("case", NO_INTERSECTION_CASES, ids=lambda case: case.name)
def test_no_intersection_meshing(case: IntersectionCase) -> None:
    """Test meshing for cases where the well does not intersect any fracture."""

    _assert_intersection_meshing(case, _expected_meshing_intersections(case))


@pytest.mark.parametrize("case", MULTI_WELL_CASES, ids=lambda case: case.name)
def test_multi_well_meshing(case: IntersectionCase) -> None:
    """Test meshing for cases where multiple wells intersect the fracture network."""

    _assert_intersection_meshing(case, _expected_meshing_intersections(case))


@pytest.mark.parametrize("case", SHARED_INTERSECTION_CASES, ids=lambda case: case.name)
def test_shared_intersection_meshing(case: IntersectionCase) -> None:
    """Test meshing for cases where one well hits a shared fracture intersection."""

    expected = _expected_meshing_intersections(case)
    if len(expected[0][2]) > 2:
        with pytest.raises(ValueError):
            _assert_intersection_meshing(case, _expected_meshing_intersections(case))
    else:
        _assert_intersection_meshing(case, expected)


def test_2d_multiple_fractures_same_point_meshing() -> None:
    """Test meshing for a 2D well crossing a shared fracture intersection point."""

    case = MULTIPLE_FRACTURES_SAME_POINT_2D_CASES
    with pytest.raises(ValueError):
        _assert_intersection_meshing(case, _expected_meshing_intersections(case))
