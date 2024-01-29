"""Testing functionality related to FractureNetwork3d.

Content:
    - Tests of the methods to impose an external boundary to the domain.
    - Test of the functionality to determine the mesh size.
    - Test of meshing.

"""
from collections import namedtuple

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs.plane_fracture import PlaneFracture


@pytest.mark.parametrize(
    "points_expected",
    [
        # Completely outside lower.
        {
            "points": [
                [-2.0, -1.0, -1.0, -2.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 1.0, 1.0],
            ],
            "expected_num_fractures": 0,
        },
        # Outside west bottom.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [-1.5, -1.5, -0.5, -0.5],
            ],
            "expected_num_fractures": 0,
        },
        # Intersect one.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_points": [
                [0.0, 0.5, 0.5, 0],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_num_fractures": 1,
        },
        # Full incline.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 1.5, 1.5],
                [-0.5, -0.5, 1, 1],
            ],
            "expected_points": [
                [0.0, 0.5, 0.5, 0],
                [5.0 / 6, 5.0 / 6, 1, 1],
                [0.0, 0.0, 0.25, 0.25],
            ],
            "expected_num_fractures": 1,
        },
        # Incline in plane.
        {
            "points": [
                [-0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, -0.5, 0.5, 1.0],
            ],
            "expected_points": [
                [0.0, 0.5, 0.5, 0],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.75],
            ],
            "expected_num_fractures": 1,
        },
        # Intersect two same.
        {
            "points": [
                [-0.5, 1.5, 1.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_points": [
                [0.0, 1, 1, 0],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.2, 0.8, 0.8],
            ],
            "expected_num_fractures": 1,
        },
    ],
)
def test_impose_external_boundary(points_expected):
    """Test of algorithm for constraining a fracture a bounding box.

    Since that algorithm uses fracture intersection methods, the tests functions as
    partial test for the wider fracture intersection framework as well. Full tests
    of the latter are too time consuming to fit into a unit test.

    Now the boundary is defined as set of "fake" fractures, all fracture network
    have 2*dim additional fractures (hence the + 6 in the assertions)

    """

    points = np.array(points_expected["points"])
    expected_num_fractures = points_expected["expected_num_fractures"]
    fracture = pp.PlaneFracture(points, check_convexity=False)
    network = pp.create_fracture_network([fracture])
    fractures_kept, fractures_deleted = network.impose_external_boundary(unit_domain(3))
    assert len(fractures_kept) == expected_num_fractures
    assert len(fractures_deleted) == 1 - expected_num_fractures
    assert len(network.fractures) == (6 + expected_num_fractures)
    p_comp = network.fractures[0].pts
    if expected_points := points_expected.get("expected_points", None):
        assert compare_arrays(p_comp, np.array(expected_points), sort=True, tol=1e-5)


def test_impose_external_boundary_bounding_box():
    # Test of method FractureNetwork.bounding_box() when an external
    # boundary is added. Thus double as test of this adding.
    fracture = PlaneFracture(
        np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]]),
        check_convexity=False,
    )
    network = pp.create_fracture_network([fracture])

    external_box = {"xmin": -1, "xmax": 2, "ymin": -1, "ymax": 2, "zmin": -1, "zmax": 2}
    domain_to_impose = pp.Domain(bounding_box=external_box.copy())
    network.impose_external_boundary(domain=domain_to_impose)

    assert network.domain.bounding_box == external_box


@pytest.mark.parametrize(
    "fracs_expected",
    [
        {
            "fracs": [
                PlaneFracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
                    check_convexity=False,
                )
            ],
            "expected": dict(xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1),
        },
        {
            "fracs": [
                PlaneFracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]),
                    check_convexity=False,
                )
            ],
            "expected": dict(xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=0),
        },
        # Test two fractures
        {
            "fracs": [
                PlaneFracture(
                    np.array([[0, 2, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
                    check_convexity=False,
                ),
                PlaneFracture(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, -1, 1, 1]]),
                    check_convexity=False,
                ),
            ],
            "expected": dict(xmin=0, xmax=2, ymin=0, ymax=1, zmin=-1, zmax=1),
        },
    ],
)
def test_bounding_box(fracs_expected):
    # Test of method FractureNetwork.bounding_box() to inquire about network extent.
    fracs = fracs_expected["fracs"]
    expected = fracs_expected["expected"]
    network = pp.create_fracture_network(fracs)
    d = network.bounding_box()
    assert d == expected


def test_mesh_size_determination():
    """3d domain. One fracture which extends to the boundary.

    Check that the mesh size functionality in FractureNetwork3d determines the expected
    mesh sizes by comparing against hard-coded values.

    """

    f_1 = np.array([[1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
    f_set = [pp.PlaneFracture(f_1)]
    domain = pp.Domain(
        {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 5, "ymax": 5, "zmax": 5}
    )
    on_boundary = np.array(
        [False, False, False, False, True, True, True, True, True, True, True, True]
    )
    # Mesh size arguments
    mesh_size_min = 0.1
    mesh_size_frac = 0.1
    mesh_size_bound = 2
    # Define a fracture network, impose the boundary, find and split intersections.
    # These operations mirror those performed in FractureNetwork3d.mesh() up to the
    # point where the mesh size is determined and auxiliary points used to enforce the
    # mesh size are inserted.
    network = pp.create_fracture_network(f_set)
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network._insert_auxiliary_points(
        mesh_size_frac=mesh_size_frac,
        mesh_size_min=mesh_size_min,
        mesh_size_bound=mesh_size_bound,
    )
    mesh_size = network._determine_mesh_size(point_tags=on_boundary)

    # To get the mesh size, we need to access the decomposition of the domain.
    decomp = network.decomposition

    # Many of the points should have mesh size of 2, adjust the exceptions below.
    mesh_size_known = np.full(decomp["points"].shape[1], mesh_size_bound, dtype=float)

    # Find the points on the fracture
    fracture_poly = np.where(np.logical_not(network.tags["boundary"]))[0]
    # There is only one fracture. This is a sanity check.
    assert fracture_poly.size == 1

    # Points on the fracture should have been assigned fracture mesh size.
    fracture_points = decomp["polygons"][fracture_poly[0]][0]
    mesh_size_known[fracture_points] = mesh_size_frac

    # Two of the domain corners are close enough to the fracture to have their mesh size
    # modified from the default boundary value.
    origin = np.zeros((3, 1))
    _, ind = pp.utils.setmembership.ismember_rows(origin, decomp["points"])
    mesh_size_known[ind] = np.sqrt(3)

    corner = np.array([5, 0, 0]).reshape((3, 1))
    _, ind = pp.utils.setmembership.ismember_rows(corner, decomp["points"], sort=False)
    mesh_size_known[ind] = np.sqrt(2)

    assert np.all(np.isclose(mesh_size, mesh_size_known))


# Named tuple used to identify intersections of fractures by their parent fractures
# and their coordinates. The coordinates should describe the full intersection line,
# independent of how many 1d grids are created from it.
IntersectionInfo = namedtuple("IntersectionInfo", ["parent_0", "parent_1", "coord"])
IntersectionInfo3Frac = namedtuple(
    "IntersectionInfo3Frac", ["parent_0", "parent_1", "parent_2", "coord"]
)


def _standard_domain(modify: bool = False) -> dict | pp.Domain:
    """Create a standard domain for testing purposes."""
    bbox = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
    if modify:
        return bbox
    else:
        domain = pp.Domain(bbox)
        return domain


def _create_mdg(
    fractures, domain=None, mesh_args: dict | None = None, constraints=None
) -> pp.MixedDimensionalGrid:
    """Create a mixed-dimensional grid from a list of fractures."""
    if mesh_args is None:
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
    network = pp.create_fracture_network(fractures, domain=domain)
    if constraints is None:
        mdg = network.mesh(mesh_args)
    else:
        mdg = network.mesh(mesh_args, constraints=constraints)
    return mdg


def check_mdg(
    mdg: pp.MixedDimensionalGrid,
    domain: pp.Domain,
    fractures=None,
    isect_line=None,
    isect_pt=None,
    expected_num_1d_grids=0,
    expected_num_0d_grids=0,
):
    """Validate the generated mixed-dimensional grid based on the expected grid properties.

    Parameters:
        mdg: Grid to be validated
        domain: Domain of the grid
        fractures: List of fractures in the domain
        isect_line: List of expected intersection lines
        isect_pt: List of expected intersection points
        expected_num_1d_grids: Expected number of 1d grids
        expected_num_0d_grids: Expected number of 0d grids

    """
    if fractures is None:
        fractures = []
    if isect_line is None:
        isect_line = []
    if isect_pt is None:
        isect_pt = []

    def compare_bounding_boxes(box_1, box_2):
        # Helper method to compare two bounding boxes
        for k, v in box_1.items():
            if np.abs(box_2[k] - v) > 1e-10:
                return False
        return True

    # Get the bounding box of the nodes of the 3d grid.
    bb = pp.domain.bounding_box_of_point_cloud(mdg.subdomains(dim=3)[0].nodes)
    assert compare_bounding_boxes(bb, domain.bounding_box)

    # Check that the number of subdomain grids are as expected
    assert len(fractures) == len(mdg.subdomains(dim=2))
    assert expected_num_1d_grids == len(mdg.subdomains(dim=1))
    assert expected_num_0d_grids == len(mdg.subdomains(dim=0))

    # Loop over all fractures, find the grid with the corresponding frac_num. Check
    # that their bounding boxes are the same.
    for fi, f in enumerate(fractures):
        for sd in mdg.subdomains(dim=2):
            if sd.frac_num == fi:
                assert compare_bounding_boxes(
                    pp.domain.bounding_box_of_point_cloud(f.pts),
                    pp.domain.bounding_box_of_point_cloud(sd.nodes),
                )

    # The bounding boxes of the constructed 1d grids will be compared to the expected
    # values. To construct the grid bounding boxes, we will loop over all 1d grids and
    # update the bounding box of the corresponding intersection grid.

    # Bounding box for the computed intersection grids. The key is a tuple of the
    # parent fractures, the value is a dictionary with the bounding box and the
    # intersection line which should result from this combination of fractures.
    ii_computed_box: dict[tuple[int, int] | tuple[int, int, int], dict] = {}

    for isect in isect_line:
        # The initial assumption is that the bounding box is empty, signified by min and
        # max values being inf and -inf, respectively (meaningful values will be
        # inserted once we start updating the boxes).
        inital_box = {
            "xmin": np.inf,
            "xmax": -np.inf,
            "ymin": np.inf,
            "ymax": -np.inf,
            "zmin": np.inf,
            "zmax": -np.inf,
        }
        if isinstance(isect, IntersectionInfo):
            # This is an intersection of two fractures. Sort the parents to get a
            # unique key.
            p_0, p_1 = sorted([isect.parent_0, isect.parent_1])

            ii_computed_box[(p_0, p_1)] = {
                "isect": isect,
                "coord": inital_box,
            }
        elif isinstance(isect, IntersectionInfo3Frac):
            # This is an intersection of three fractures.
            p_0, p_1, p_2 = sorted([isect.parent_0, isect.parent_1, isect.parent_2])
            ii_computed_box[(p_0, p_1, p_2)] = {
                "coord": inital_box,
                "isect": isect,
            }

    def update_box(box, update):
        # Helper method to update a bounding box.
        for k in ["xmin", "ymin", "zmin"]:
            box["coord"][k] = min(box["coord"][k], update[k])
        for k in ["xmax", "ymax", "zmax"]:
            box["coord"][k] = max(box["coord"][k], update[k])

    # Loop over the 1d domains, and update the bounding boxes of the intersection grid.
    # Since a fracture intersection may be shared by several 1d grids (exactly how this
    # is handled is not clear to EK at the moment, but it surely cannot be wrong to
    # allow for more than one 1d grid to share an intersection), we need to update the
    # bounding box of the intersection grid for each 1d grid that shares the
    # intersection.
    for sd in mdg.subdomains(dim=1):
        # Get the parents (fractures) of the 1d grid
        neighs = mdg.neighboring_subdomains(sd, only_higher=True)
        # bounding box of the 1d grid
        box = pp.domain.bounding_box_of_point_cloud(sd.nodes)

        # Update the bounding box of the intersection grid (identified by the parent
        # fractures).
        if len(neighs) == 2:
            f_0, f_1 = sorted([neighs[0].frac_num, neighs[1].frac_num])
            update_box(ii_computed_box[(f_0, f_1)], box)

        elif len(neighs) == 3:
            f_0, f_1, f_2 = sorted(
                [neighs[0].frac_num, neighs[1].frac_num, neighs[2].frac_num]
            )
            update_box(ii_computed_box[(f_0, f_1, f_2)], box)

    # Check that the bounding boxes of the intersection grids correspond to the
    # expected values.
    for val in ii_computed_box.values():
        coord = val["coord"]
        isect = val["isect"]
        assert compare_bounding_boxes(
            coord, pp.domain.bounding_box_of_point_cloud(isect.coord)
        )

    # For each 0d grid, check that it is present as an expected intersection point.
    for sd in mdg.subdomains(dim=0):
        found = False
        for p in isect_pt:
            if np.allclose(p, sd.cell_centers):
                found = True
                break
        assert found
    # For each intersection point, check that it is present as a 0d grid.
    for p in isect_pt:
        found = False
        for sd in mdg.subdomains(dim=0):
            if np.allclose(p, sd.cell_centers):
                found = True
                break
        assert found


class TestDFMMeshGeneration:
    """Test meshing of fracture networks in 3d. No fracture hits the domain boundary."""

    def test_no_fracture(self):
        domain = _standard_domain()
        mdg = _create_mdg(None, domain)
        check_mdg(mdg, domain)

    def test_single_isolated_fracture(self):
        """A single fracture completely immersed in a boundary grid."""
        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])

    def test_two_intersecting_fractures(self):
        """Two fractures intersecting along a line."""

        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])

        isect = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect],
            expected_num_1d_grids=1,
        )

    def test_three_intersecting_fractures(self):
        """Three fractures intersecting, with intersecting intersections (point)"""

        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        f_3 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]]))

        # Add some parameters for grid size
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain)

        isect_0_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])
        isect_1_coord = np.array([[-1, 1], [0, 0], [0, 0]])
        isect_2_coord = np.array([[0, 0], [-1, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_0_coord)
        isect_1 = IntersectionInfo(0, 2, isect_1_coord)
        isect_2 = IntersectionInfo(1, 2, isect_2_coord)

        isect_pt = np.array([0, 0, 0]).reshape((-1, 1))

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect_0, isect_1, isect_2],
            isect_pt=[isect_pt],
            expected_num_1d_grids=6,
            expected_num_0d_grids=1,
        )

    def test_one_fracture_intersected_by_two(self):
        """One fracture, intersected by two other (but no point intersections)."""

        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        f_3 = pp.PlaneFracture(f_2.pts + np.array([0.5, 0, 0]).reshape((-1, 1)))

        # Add some parameters for grid size
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain)

        isect_0_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])

        isect_1_coord = np.array([[0.5, 0.5], [0, 0], [-0.7, 0.8]])

        isect_0 = IntersectionInfo(0, 1, isect_0_coord)
        isect_1 = IntersectionInfo(0, 2, isect_1_coord)
        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect_0, isect_1],
            expected_num_1d_grids=2,
        )

    def test_split_into_octants(self):
        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [0, 0, 0, 0]]))
        f_3 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = _standard_domain()

        mdg = _create_mdg([f_1, f_2, f_3], domain)

        isect_0_coord = np.array([[0, 0], [0, 0], [-1, 1]])
        isect_1_coord = np.array([[-1, 1], [0, 0], [0, 0]])
        isect_2_coord = np.array([[0, 0], [-1, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 2, isect_0_coord)
        isect_1 = IntersectionInfo(0, 1, isect_1_coord)
        isect_2 = IntersectionInfo(1, 2, isect_2_coord)

        isect_pt = np.array([0, 0, 0]).reshape((-1, 1))

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect_0, isect_1, isect_2],
            isect_pt=[isect_pt],
            expected_num_1d_grids=6,
            expected_num_0d_grids=1,
        )

    def test_three_fractures_sharing_line_same_segment(self):
        """
        Three fractures that all share an intersection line. This can be considered
        as three intersection lines that coincide.
        """
        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        )
        f_3 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain)

        isect_coord = np.array([[0, 0], [0, 0], [-1, 1]])
        isect = IntersectionInfo3Frac(0, 1, 2, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect],
            expected_num_1d_grids=1,
            expected_num_0d_grids=0,
        )

    def test_three_fractures_split_segments(self):
        """
        Three fractures that all intersect along the same line, but with the
        intersection between two of them forming an extension of the intersection
        of all three.
        """
        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-0.5, -0.5, 0.5, 0.5]])
        )
        f_3 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))

        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain)

        # Fractures 0 and 2 intersect both in the interval z\in[-1, -0.5] and
        # [0.5, 1]. The test for intersection lines is not smart enough for this,
        # so we will simply define a full line of [-1, 1]. This is not ideal, but
        # better than nothing.
        isect_coord_full = np.array([[0, 0], [0, 0], [-1, 1]])
        isect_0 = IntersectionInfo(0, 2, isect_coord_full)

        # The intersection between three lines can be handled properly.
        isect_coord_middle = np.array([[0, 0], [0, 0], [-0.5, 0.5]])
        isect_1 = IntersectionInfo3Frac(0, 1, 2, isect_coord_middle)

        isect_pt = [
            np.array([0, 0, -0.5]).reshape((-1, 1)),
            np.array([0, 0, 0.5]).reshape((-1, 1)),
        ]

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect_0, isect_1],
            isect_pt=isect_pt,
            # Note that we expect three intersection grids here
            expected_num_1d_grids=3,
            expected_num_0d_grids=2,
        )

    def test_two_fractures_L_intersection(self):
        """Two fractures sharing a segment in an L-intersection."""
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))

        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0, 0], [0, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_two_fractures_L_intersection_part_of_segment(self):
        """Two fractures sharing what is a full segment for one and part of a segment
        for the other.
        """
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]])
        )
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0, 0], [0.3, 0.7], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_two_fractures_L_intersection_one_displaced(self):
        """Two fractures sharing what is a part of segments for both."""
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [0.5, 1.5, 1.5, 0.5], [0, 0, 1, 1]])
        )
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0, 0], [0.5, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)
        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_T_intersection_within_plane(self):
        """Two fractures intersecting in a T-style intersection.

        One of the edges of the second fracture is immersed in the first fracture.
        """
        f_1 = pp.PlaneFracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 0.9, 0.0]]).T
        )

        # Modify the domain.
        bbox = _standard_domain(modify=True)
        bbox["xmin"] = -1
        bbox["zmin"] = -1
        domain = pp.Domain(bbox)
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.5, 0.5], [0.5, 0.9], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_T_intersection_one_outside_plane(self):
        """Two fractures intersecting in a T-style intersection.

        The edge of the second fracture lying in the first fracture plane extends
        outside the first fracture on one side.

        """
        f_1 = pp.PlaneFracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 1.9, 0.0]]).T
        )

        bbox = _standard_domain(modify=True)
        bbox["xmin"] = -1
        bbox["zmin"] = -1
        domain = pp.Domain(bbox)
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.5, 0.5], [0.5, 1.0], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_T_intersection_both_outside_plane(self):
        """Two fractures intersecting in a T-style intersection.

        The edge of the second fracture lying in the first fracture plane extends
        outside the first fracture on both sides.

        """
        f_1 = pp.PlaneFracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 1], [0.5, -0.5, 0], [0.5, 1.9, 0.0]]).T
        )

        bbox = _standard_domain(modify=True)
        bbox["xmin"] = -1
        bbox["zmin"] = -1
        domain = pp.Domain(bbox)
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.5, 0.5], [0.0, 1.0], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_T_intersection_both_on_boundary(self):
        """T-style intersection, the end of the intersection line coincides with the
        boundary of both fractures.

        To debug this test, first make a drawing of the geometry to understand the
        fracture configuration.
        """
        f_1 = pp.PlaneFracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.PlaneFracture(
            np.array([[0.0, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T
        )

        bbox = _standard_domain(modify=True)
        bbox["xmin"] = -1
        bbox["zmin"] = -1
        domain = pp.Domain(bbox)
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.0, 1.0], [0.5, 0.5], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_T_intersection_one_boundary_one_outside(self):
        """T-style intersection. The end of the intersection line coincides with the
        boundary of one fracture, but not the other.

        To debug this test, first make a drawing of the geometry to understand the
        fracture configuration.
        """
        f_1 = pp.PlaneFracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.PlaneFracture(
            np.array([[-0.2, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T
        )

        bbox = _standard_domain(modify=True)
        bbox["xmin"] = -1
        bbox["zmin"] = -1
        domain = pp.Domain(bbox)
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.0, 1.0], [0.5, 0.5], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_T_intersection_one_boundary_one_inside(self):
        """T-style intersection. One fracture has one intersection point (end of the
        intersection line) in its interior, one on the boundary.

        To debug this test, first make a drawing of the geometry to understand the
        fracture configuration.
        """
        f_1 = pp.PlaneFracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.PlaneFracture(
            np.array([[0.2, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T
        )

        bbox = _standard_domain(modify=True)
        bbox["xmin"] = -1
        bbox["zmin"] = -1
        domain = pp.Domain(bbox)
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.2, 1.0], [0.5, 0.5], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_issue_54(self):
        """This gave an error in the past, and is kept as a sanity check."""
        f_1 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [0.4, 0.5, 0.5, 0.4], [0.2, 0.2, 0.8, 0.8]])
        )
        f_2 = pp.PlaneFracture(
            np.array([[0, 0.8, 0.8, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
        )
        # Use a special domain here.
        bbox = {"xmin": -1, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        domain = pp.Domain(bbox)
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 0.8]])
        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_issue_58_1(self):
        """This gave an error in the past, and is kept as a sanity check."""
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0.5, 0.5, 0], [0, 0, 1, 1]]))
        domain = _standard_domain()

        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0, 1], [0, 0.5], [0, 0]])
        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_issue_58_2(self):
        """This gave an error in the past, and is kept as a sanity check."""
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0.3, 0.5, 0.5, 0.3], [0, 0, 1, 1]])
        )
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)
        isect_coord = np.array([[0, 1], [0.3, 0.5], [0, 0]])
        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_issue_58_3(self):
        """This gave an error in the past, and is kept as a sanity check."""
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)
        isect_coord = np.array([[0, 1], [0, 1], [0, 0]])
        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )

    def test_partial_rubics_cube(self):
        """This is a part of a rubics-cube style fracture network."""
        f_0 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        )
        f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
        )
        f_2 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        )
        f_3 = pp.PlaneFracture(
            np.array([[0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1], [0.75, 0.75, 0.75, 0.75]])
        )
        f_4 = pp.PlaneFracture(
            np.array([[0.75, 0.75, 0.75, 0.75], [0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1]])
        )

        # This test does not use the standard domain or mesh size arguments, thus we
        # do meshing by hand.
        bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        domain = pp.Domain(bbox)

        mesh_args = {
            "mesh_size_frac": 0.4,
            "mesh_size_bound": 1,
            "mesh_size_min": 0.2,
            "return_expected": True,
        }
        network = pp.create_fracture_network([f_0, f_1, f_2, f_3, f_4], domain)
        mdg = network.mesh(mesh_args)

        # Known intersection lines.
        isect_lines = [
            # Fracture 0 and 1
            IntersectionInfo(0, 1, np.array([[0.5, 0.5], [0.5, 0.5], [0, 1]])),
            # Fracture 0 and 2
            IntersectionInfo(0, 2, np.array([[0.5, 0.5], [0, 1], [0.5, 0.5]])),
            # Fracture 1 and 2
            IntersectionInfo(1, 2, np.array([[0, 1], [0.5, 0.5], [0.5, 0.5]])),
            # Fracture 0 and 3
            IntersectionInfo(0, 3, np.array([[0.5, 0.5], [0.5, 1], [0.75, 0.75]])),
            # Fracture 1 and 3
            IntersectionInfo(1, 3, np.array([[0.5, 1], [0.5, 0.5], [0.75, 0.75]])),
            # Fracture 1 and 4
            IntersectionInfo(1, 4, np.array([[0.75, 0.75], [0.5, 0.5], [0.5, 1]])),
            # Fracture 2 and 4
            IntersectionInfo(2, 4, np.array([[0.75, 0.75], [0.5, 1], [0.5, 0.5]])),
            # Fracture 3 and 4
            IntersectionInfo(3, 4, np.array([[0.75, 0.75], [0.5, 1], [0.75, 0.75]])),
        ]

        # Known intersection points.
        isect_pt = [
            np.array([0.5, 0.5, 0.75]).reshape((-1, 1)),
            np.array([0.5, 0.5, 0.5]).reshape((-1, 1)),
            np.array([0.75, 0.5, 0.75]).reshape((-1, 1)),
            np.array([0.75, 0.5, 0.5]).reshape((-1, 1)),
        ]

        check_mdg(
            mdg,
            domain,
            fractures=[f_0, f_1, f_2, f_3, f_4],
            expected_num_1d_grids=15,
            expected_num_0d_grids=4,
            isect_line=isect_lines,
            isect_pt=isect_pt,
        )

    def test_issue_90(self):
        """This gave an error in the past, and is kept as a sanity check."""
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(np.array([[0, 1, 1], [0, 0, 1], [1, 1, -1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)

        isect_coord = np.array([[0.5, 1], [0.5, 0.5], [0, 0]])
        isect_0 = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            expected_num_1d_grids=1,
        )


class TestMeshGenerationFractureHitsBoundary:
    """Test DFM meshing for networks where at least one fracture reaches the domain
    boundary.
    """

    def kwarguments_and_domain(self):
        bbox = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        domain = pp.Domain(bbox)
        mesh_args = {"mesh_size_frac": 1, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        return mesh_args, domain

    def test_one_fracture_touching_one_boundary(self):
        """A single fracture completely immersed in a boundary grid."""
        f_1 = pp.PlaneFracture(np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])

    def test_one_fracture_touching_two_neighbouring_boundaries(self):
        f_1 = pp.PlaneFracture(np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 2, 2]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])

    def test_one_fracture_touching_two_opposing_boundaries(self):
        f_1 = pp.PlaneFracture(np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])

    def test_one_fracture_touching_three_boundaries(self):
        f_1 = pp.PlaneFracture(np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-1, -1, 2, 2]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])

    def test_one_fracture_touches_four_boundaries(self):
        f_1 = pp.PlaneFracture(np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-2, -2, 2, 2]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])

    def test_one_fracture_touches_boundary_corners(self):
        f_1 = pp.PlaneFracture(
            np.array([[-2, 2, 2, -2], [-2, 2, 2, -2], [-2, -2, 2, 2]])
        )
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])

    def test_two_fractures_touching_one_boundary(self):
        f_1 = pp.PlaneFracture(np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(np.array([[-1, -1, 2, 2], [-1, 1, 1, -1], [0, 0, 0, 0]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain)
        isect_line = [IntersectionInfo(0, 1, np.array([[-1, 2], [0, 0], [0, 0]]))]
        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=isect_line,
            expected_num_1d_grids=1,
        )

    def test_one_fracture_touches_boundary_and_corner(self):
        f_1 = pp.PlaneFracture(
            np.array([[-2, 2, 2, -2], [-2, 1, 1, -2], [-2, -2, 2, 2]])
        )
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain)
        check_mdg(mdg, domain, fractures=[f_1])


class TestDFMMeshGenerationWithConstraints:
    """Tests similar to some those of TestDFMMeshGeneration, but with constraints."""

    def test_single_isolated_fracture(self):
        """
        A single fracture completely immersed in a boundary grid.
        """
        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1], domain, constraints=np.array([0]))
        check_mdg(mdg, domain)

    def test_two_intersecting_fractures(self):
        """Two fractures intersecting along a line."""

        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain, constraints=np.array([1]))
        check_mdg(mdg, domain, fractures=[f_1])

    def test_three_intersecting_fractures_one_constraint(self):
        """Three fractures intersecting, with intersecting intersections (point)."""

        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        f_3 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]]))

        # Add some parameters for grid size
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain, constraints=np.array([2]))

        isect_0_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])

        isect_0 = IntersectionInfo(0, 1, isect_0_coord)

        isect_pt = np.array([0, 0, 0]).reshape((-1, 1))

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect_0],
            isect_pt=[isect_pt],
            expected_num_1d_grids=2,
            expected_num_0d_grids=1,
        )

    def test_three_fractures_sharing_segment_one_constraint(self):
        """
        Three fractures that all share an intersection line. This can be considered
        as three intersection lines that coincide.
        """
        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        )
        f_3 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain, constraints=np.array([2]))

        isect_coord = np.array([[0, 0], [0, 0], [-1, 1]])
        isect = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect],
            expected_num_1d_grids=1,
            expected_num_0d_grids=0,
        )

    def test_three_fractures_sharing_segment_two_constraints(self):
        """
        Three fractures that all share an intersection line. This can be considered
        as three intersection lines that coincide.
        """
        f_1 = pp.PlaneFracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
        )
        f_3 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain, constraints=np.array([1, 2]))

        check_mdg(
            mdg,
            domain,
            fractures=[f_1],
            expected_num_1d_grids=0,
            expected_num_0d_grids=0,
        )

    def test_two_fractures_L_intersection_one_constraint(self):
        """
        Two fractures sharing a segment in an L-intersection.
        """
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))
        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2], domain, constraints=np.array([1]))
        check_mdg(mdg, domain, fractures=[f_1])

    def test_T_intersection_is_also_within_constraint(self):
        # a fracture
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        # a fracture that abuts the first fracture
        f_2 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        )
        # a fracture, to be constraint, that intersects in the same line
        f_3 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, 1, 1, -1]]))

        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain, constraints=np.array([2]))

        isect_coord = np.array([[0.5, 0.5], [0, 1], [0, 0]])
        isect = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect],
            expected_num_1d_grids=1,
            expected_num_0d_grids=0,
        )

    def test_T_intersection_is_partially_within_constraint(self):
        # a fracture
        f_1 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        # a fracture that abuts the first fracture
        f_2 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [0.3, 1, 1, 0], [0, 0, 1, 1]])
        )
        # a fracture, to be constraint, that intersects in the same line
        f_3 = pp.PlaneFracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, 1, 1, -1]]))

        domain = _standard_domain()
        mdg = _create_mdg([f_1, f_2, f_3], domain, constraints=np.array([2]))

        isect_coord = np.array([[0.5, 0.5], [0.3, 1], [0, 0]])
        isect = IntersectionInfo(0, 1, isect_coord)

        check_mdg(
            mdg,
            domain,
            fractures=[f_1, f_2],
            isect_line=[isect],
            expected_num_1d_grids=1,
            expected_num_0d_grids=0,
        )


class TestDFNMeshGeneration:
    def _generate_mesh(self, fractures):
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}
        network = pp.create_fracture_network(fractures)
        mdg = network.mesh(mesh_args, dfn=True)
        return mdg

    def test_conforming_two_fractures(self):
        f_1 = pp.PlaneFracture(
            np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T
        )
        f_2 = pp.PlaneFracture(
            np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T
        )
        mdg = self._generate_mesh([f_1, f_2])
        assert len(mdg.subdomains(dim=2)) == 2
        assert len(mdg.subdomains(dim=1)) == 1

    def test_conforming_three_fractures(self):
        f_1 = pp.PlaneFracture(
            np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T
        )
        f_2 = pp.PlaneFracture(
            np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T
        )
        f_3 = pp.PlaneFracture(
            np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]).T
        )
        mdg = self._generate_mesh([f_1, f_2, f_3])
        assert len(mdg.subdomains(dim=2)) == 3
        assert len(mdg.subdomains(dim=1)) == 6
        assert len(mdg.subdomains(dim=0)) == 1


class TestDFMNonConvexDomain:
    """Test fracture meshing in non-convex domains.

    NOTE: The corresponding source code is complex, and the test suite below does not
    cover all cases.

    """

    def domain(self):
        """Set up a non-convex domain."""
        west = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        south_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 0], [0, 0.5, 1, 1]])
        south_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 0, 0], [0.5, 0, 1, 1]])
        north_w = np.array([[0, 0.5, 0.5, 0], [1, 1, 1, 1], [0, 0.5, 1, 1]])
        north_e = np.array([[0.5, 1, 1, 0.5], [1, 1, 1, 1], [0.5, 0, 1, 1]])
        bottom_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [0, 0.5, 0.5, 0]])
        bottom_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [0.5, 0.0, 0, 0.5]])
        top_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        top_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [1, 1, 1, 1]])
        return [
            west,
            east,
            south_w,
            south_e,
            north_w,
            north_e,
            bottom_w,
            bottom_e,
            top_w,
            top_e,
        ]

    def _generate_mesh(self, fractures):
        domain = pp.Domain(polytope=self.domain())
        network = pp.create_fracture_network(fractures, domain)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        mdg = network.mesh(mesh_args)
        return mdg

    def test_fracture_split_by_domain(self):
        """The fracture should be split into subfractures because of the non-convexity
        of the domain.
        """

        f_1 = pp.PlaneFracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.3, 0.3]])
        )
        mdg = self._generate_mesh([f_1])
        assert len(mdg.subdomains(dim=2)) == 2

    def test_fracture_cut_not_split_by_domain(self):
        f_1 = pp.PlaneFracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.7, 0.7]])
        )
        mdg = self._generate_mesh([f_1])
        # The fracture should be split into subfractures because of the non-convexity.
        # non-convexity. The size of the domain here is set by how the fracture pieces
        # are merged into convex parts.
        # EK comment/ interpretation (written long after the test was written): The
        # comment above is opaque, but relates to how non-convex fractures / domains are
        # handled in FractureNetwork3d.mesh(). This is a particularly complex part of
        # the code (and that says something), but it is also something which is not used
        # a lot. For now, we keep the test, interpreted mainly as a check that the code
        # functions as before. Should we ever clean up the accompanying code, this test
        # should be rewritten.
        assert len(mdg.subdomains(dim=2)) == 3
