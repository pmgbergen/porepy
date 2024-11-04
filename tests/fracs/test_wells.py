"""
Tests of the well class. In particular, functionality for constructing the
well network and the resulting updates to the mixed-dimensional grid are tested.

Content:
  * Addition of one well to mdgs with one or two 2d fractures.
  * Addition of two wells to mdgs with one or three 2d fractures.
Both tests check for number of grids, number of edges and three types of face
tags. Grid node ordering is tacitly assumed - if the assumption is broken, the
well implementation should also be revisited.

"""
import numbers
from typing import List

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain


@pytest.fixture
def get_mdg():
    def inner(fracture_indices: List[int], well_indices: List[int]):
        """Construct networks and generate mdg.

        Parameters:
            fracture_indices: which fractures to use.
            well_indices: which wells to use.

        Returns:
            Mixed-dimensional grid with matrix, fractures, wells and
            well-fracture intersection grids + all interfaces

        """

        # Three horizontal fractures
        fracture_coords = [
            np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]]),
            np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.2, 0.2, 0.2, 0.2]]),
            np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.1, 0.1, 0.1, 0.1]]),
        ]
        fractures = [pp.PlaneFracture(fracture_coords[i]) for i in fracture_indices]
        fracture_network = pp.create_fracture_network(fractures, unit_domain(3))

        # Vertical well extending from 0.1 (frac 2) to upper boundary and
        #   tilted well extending from 0.2 (frac 1) to upper boundary
        well_coords = [
            np.array([[0.5, 0.5], [0.5, 0.5], [1, 0.1]]),
            np.array([[0.5, 0.6], [0.7, 0.8], [1, 0.2]]),
        ]
        wells = [pp.Well(well_coords[i]) for i in well_indices]
        well_network = pp.WellNetwork3d(
            unit_domain(3), wells, parameters={"mesh_size": 1}
        )

        mdg = fracture_network.mesh({"mesh_size_frac": 1, "mesh_size_min": 1})

        # Compute intersections
        pp.fracs.wells_3d.compute_well_fracture_intersections(
            well_network, fracture_network
        )
        # Mesh fractures and add fracture + intersection grids to the md-grid
        # along with these grids' new interfaces to fractures.
        well_network.mesh(mdg)

        return mdg

    return inner


@pytest.mark.parametrize(
    "fracture_indices, fracture_faces, tip_faces",
    [
        ([0], [[0, 1], [1, 0]], [[0, 0], [0, 1]]),  # Single internal
        ([2], [[0, 1], [1, 0]], [[0, 0], [0, 0]]),  # Single at well endpoint
        ([1, 0], [[0, 1], [1, 1], [1, 0]], [[0, 0], [0, 0], [0, 1]]),  # Two internal
        ([0, 2], [[0, 1], [1, 1]], [[0, 0], [0, 0]]),  # Internal and endpoint
    ],
)
def test_add_one_well(
    fracture_indices: List[int],
    fracture_faces: List[List[int]],
    tip_faces: List[List[int]],
    request,
) -> None:
    """Compute intersection between one well and the fracture network, mesh and
    add well grids to mdg.

    Parameters:
        fracture_indices: which fractures to use.
        fracture_faces: Each item is the expected fracture face tags for one
            well grid, assumed to have two faces each.
        tip_faces: Each item is the expected tip face tags for one well grid,
            assumed to have two faces each.

    """
    mdg = request.getfixturevalue("get_mdg")(fracture_indices, [0])
    # One 3d grid, n_frac 2d grids, n_frac 1d well grids + one if none of the
    # fractures are on the well endpoint and n_frac intersections between
    # fractures and well
    n_int = (0 in fracture_indices) + (1 in fracture_indices)
    n_end = 2 in fracture_indices
    n_frac = n_int + n_end
    assert mdg.num_subdomains() == (1 + 3 * n_frac + (1 - n_end))

    # 3d-2d: n_frac between matrix and fractures,
    # 2d-0d: n_frac
    # 1d-0d: 2 between well and intersection for each internal fracture and 1 for
    # endpoint fracture
    assert mdg.num_interfaces() == (n_frac + n_frac + 2 * n_int + n_end)

    # Only the first well grid should be on the global boundary
    boundary_faces = [[1, 0], [0, 0], [0, 0]]
    for ind, well_grid in enumerate(mdg.subdomains(dim=1)):
        assert np.all(np.isclose(well_grid.tags["fracture_faces"], fracture_faces[ind]))
        assert np.all(np.isclose(well_grid.tags["tip_faces"], tip_faces[ind]))
        assert np.all(
            np.isclose(well_grid.tags["domain_boundary_faces"], boundary_faces[ind])
        )


# Single fracture: internal to well 0 and tip for well 1.
# First two well grids (first dimension below) correspond to well 0, the last grid to
# well 1.
f_tags_0 = [[0, 1], [1, 0], [0, 1]]
t_tags_0 = [[0, 0], [0, 1], [0, 0]]
b_tags_0 = [[1, 0], [0, 0], [1, 0]]
# Number of grids and intersections, numbers in sums sorted by descending dimension.
# Grids: One 3d, 1 fracture, 2 + 1 well grids and 1 + 1 intersections.
# Interfaces: 1 3d-2d, 2+1 well-fracture 1+1 fracture-intersection
mdg_data_0 = [1 + 1 + 3 + 2, 1 + 3 + 2]

# All three fractures. frac 2 only intersects well 0
# First three well grids (first dimension below) correspond to well 0,
# the last three grids to well 1
f_tags_1 = [[0, 1], [1, 1], [1, 1], [0, 1], [1, 1]]
t_tags_1 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
b_tags_1 = [[1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]
# Number of grids and intersections, numbers in sums sorted by descending dimension.
# Grids: One 3d, 3 fracture, 3 + 2 well grids and 3 + 2 intersections.
# Interfaces: 3 3d-2d, 3+2 well-fracture and 5+3 fracture-intersection
mdg_data_1 = [1 + 3 + 5 + 5, 3 + 5 + 8]


@pytest.mark.parametrize(
    "fracture_indices, fracture_faces, tip_faces, boundary_faces, mdg_data",
    [
        ([1], f_tags_0, t_tags_0, b_tags_0, mdg_data_0),
        ([0, 1, 2], f_tags_1, t_tags_1, b_tags_1, mdg_data_1),
    ],
)
def test_add_two_wells(
    fracture_indices: List[int],
    fracture_faces: List[List[int]],
    tip_faces: List[List[int]],
    boundary_faces: List[List[int]],
    mdg_data: List[int],
    request,
) -> None:
    """Compute intersection between two well and the fracture network, mesh and
    add well grids to mdg.

    Parameters:
        fracture_indices: which fractures to use.
        fracture_faces: Each item is the expected fracture face tags for one
            well grid, assumed to have two faces each.
        tip_faces: Each item is the expected tip face tags for one well grid,
            assumed to have two faces each.
        boundary_faces: Each item is the expected boundary face tags for one
            well grid, assumed to have two faces each.
        mdg_data: expected number of grids and number of interfaces.

    """
    mdg = request.getfixturevalue("get_mdg")(fracture_indices, [0, 1])
    assert np.isclose(mdg.num_subdomains(), mdg_data[0])
    assert np.isclose(mdg.num_interfaces(), mdg_data[1])

    # Only the first well grid should be on the global boundary
    for ind, well_grid in enumerate(mdg.subdomains(dim=1)):
        assert np.all(np.isclose(well_grid.tags["fracture_faces"], fracture_faces[ind]))
        assert np.all(np.isclose(well_grid.tags["tip_faces"], tip_faces[ind]))
        assert np.all(
            np.isclose(well_grid.tags["domain_boundary_faces"], boundary_faces[ind])
        )


def test_add_one_well_with_matrix(get_mdg) -> None:
    """Compute intersection between one well and the rock matrix mesh."""
    mdg = get_mdg([], [1])
    # add the coupling between the rock matrix and the well
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    # check the number of subdomains and interfaces
    assert mdg.num_subdomains() == 2
    assert mdg.num_interfaces() == 1

    # check the well grid
    for well_grid in mdg.subdomains(dim=1):
        assert well_grid.num_cells == 1
        assert well_grid.num_faces == 2
        assert well_grid.num_nodes == 2

    for intf in mdg.interfaces():
        assert intf.num_sides() == 1
        assert intf.num_cells == 1
        assert np.allclose(intf.mortar_to_secondary_int().todense(), 1)

        known = np.zeros(24)
        known[0] = 0.175
        known[3] = 0.29166667
        known[11] = 0.25
        known[22] = 0.08333333
        known[23] = 0.2

        # Since the generation of .msh files is platform-dependent, only norm values are
        # compared.
        assert np.isclose(
            np.linalg.norm(known),
            np.linalg.norm(intf.mortar_to_primary_int().toarray().flatten()),
            rtol=1e-5,
            atol=1e-8,
        )

    # Adding a well also adds a new boundary grid. Check that new boundary grid is
    # initialized.
    well_boundaries = mdg.boundaries(dim=0)
    assert len(well_boundaries) == 1
    for well_bg in well_boundaries:
        # num_cells is one of the attributes that are initialized lazily.
        assert isinstance(well_bg.num_cells, numbers.Integral)
