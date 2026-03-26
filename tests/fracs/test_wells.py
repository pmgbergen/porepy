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

        # Three horizontal fractures. The middle one is elliptic, just for the sake of
        # it.
        all_fractures = [
            pp.PlaneFracture(
                np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]])
            ),
            pp.EllipticFracture(np.array([[0.5], [0.5], [0.2]]), 0.5, 0.5, 0, 0, 0),
            pp.PlaneFracture(
                np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.1, 0.1, 0.1, 0.1]])
            ),
        ]
        # Use these fractures.
        fractures = [all_fractures[i] for i in fracture_indices]
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

        mdg = fracture_network.mesh(
            mesh_args={
                "mesh_size_fracture": 1,
                "mesh_size_boundary": 1,
                "mesh_size_min": 1,
                "refinement_proximity_multiplier": 1,
                "refinement_size_multiplier": 1,
                "background_transition_multiplier": 1.01,
            }
        )

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


# Single fracture: internal to well 0 and tip for well 1. The outer dictionaries map
# well number (0 or 1) to inner dictionaries. The inner dictionaries map z-coordinates
# to expected face tags. To see the logic of the tags, see the coordinate definitions
# (of both fractures and wells) in the get_mdg fixture.
f_tags_0 = {0: {0.1: False, 0.2: True, 1: False}, 1: {0.2: True, 1: False}}
t_tags_0 = {0: {0.1: True, 0.2: False, 1: False}, 1: {0.2: False, 1: False}}
b_tags_0 = {0: {0.1: False, 0.2: False, 1: True}, 1: {0.2: False, 1: True}}

# Number of grids and intersections, numbers in sums sorted by descending dimension.
# Grids: One 3d, 1 fracture, 2 + 1 well grids and 1 + 1 intersections.
# Interfaces: 1 3d-2d, 2+1 well-fracture 1+1 fracture-intersection
mdg_data_0 = [1 + 1 + 3 + 2, 1 + 3 + 2]

# All three fractures. frac 2 only intersects well 0
f_tags_1 = {
    0: {0.1: True, 0.2: True, 0.5: True, 1: False},
    1: {0.2: True, 0.5: True, 1: False},
}
t_tags_1 = {
    0: {0.1: False, 0.2: False, 0.5: False, 1: False},
    1: {0.2: False, 0.5: False, 1: False},
}
b_tags_1 = {
    0: {0.1: False, 0.2: False, 0.5: False, 1: True},
    1: {0.2: False, 0.5: False, 1: True},
}

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

    for well_grid in mdg.subdomains(dim=1):
        well_num = well_grid.tags["parent_well_index"]
        # Loop over the two endpoints of the well grid, fetch the z-coordinate to
        # identify which known endpoint (either fracture intersection or end of the full
        # well). Use this to index into the expected tags.
        for fi, z_ind in enumerate([0, -1]):
            z_coord = round(float(well_grid.nodes[2, z_ind]), ndigits=1)
            assert z_coord in fracture_faces[well_num]
            assert np.all(
                np.isclose(
                    well_grid.tags["fracture_faces"][fi],
                    fracture_faces[well_num][z_coord],
                )
            )
            assert np.all(
                np.isclose(
                    well_grid.tags["tip_faces"][fi], tip_faces[well_num][z_coord]
                )
            )
            assert np.all(
                np.isclose(
                    well_grid.tags["domain_boundary_faces"][fi],
                    boundary_faces[well_num][z_coord],
                )
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
