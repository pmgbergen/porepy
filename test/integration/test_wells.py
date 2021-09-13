"""
Tests of the well class. In particular, functionality for constructing the
well network and the resulting updates to the grid bucket are tested.

Content:
  * Addition of one well to gbs with one or two 2d fractures.
  * Addition of two wells to gbs with one or three 2d fractures.
Both tests check for number of grids, number of edges and three types of face
tags. Grid node ordering is tacitly assumed - if the assumption is broken, the
well implementation should also be revisited.
"""
import numpy as np
import pytest

import porepy as pp

from typing import List


def _generate_gb(fracture_indices: List[int], well_indices: List[int]):
    """Construct networks and generate gb.

    Parameters:
        fracture_indices (list): which fractures to use.
        well_indices (list): which wells to use.

    Returns:
        pp.GridBucket: grid bucket with matrix, fractures, wells and well-fracture
            intersection grids + all interfaces
    """
    domain = pp.grids.standard_grids.utils.unit_domain(3)

    # Three horizontal fractures
    fracture_coords = [
        np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]]),
        np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.2, 0.2, 0.2, 0.2]]),
        np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0.1, 0.1, 0.1, 0.1]]),
    ]
    fractures = [pp.Fracture(fracture_coords[i]) for i in fracture_indices]
    fracture_network = pp.FractureNetwork3d(fractures, domain)

    # Vertical well extending from 0.1 (frac 2) to upper boundary and
    #   tilted well extending from 0.2 (frac 1) to upper boundary
    well_coords = [
        np.array([[0.5, 0.5], [0.5, 0.5], [1, 0.1]]),
        np.array([[0.5, 0.6], [0.7, 0.8], [1, 0.2]]),
    ]
    wells = [pp.Well(well_coords[i]) for i in well_indices]
    well_network = pp.WellNetwork3d(wells, domain, parameters={"mesh_size": 1})

    gb = fracture_network.mesh({"mesh_size_frac": 1, "mesh_size_min": 1})

    # Compute intersections
    pp.fracs.wells_3d.compute_well_fracture_intersections(
        well_network, fracture_network
    )
    # Mesh fractures and add fracture + intersection grids to grid bucket along
    # with these grids' new interfaces to fractures.
    well_network.mesh(gb)
    return gb


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
) -> None:
    """Compute intersection between one well and the fracture network, mesh and
    add well grids to gb.

    Parameters:
        fracture_indices (list): which fractures to use.
        fracture_faces (list): Each item is the expected fracture face tags for one
            well grid, assumed to have two faces each.
        tip_faces (list): Each item is the expected tip face tags for one well grid,
            assumed to have two faces each.
    """
    gb = _generate_gb(fracture_indices, [0])
    # One 3d grid, n_frac 2d grids, n_frac 1d well grids + one if none of the
    # fractures are on the well endpoint and n_frac intersections between
    # fractures and well
    n_int = (0 in fracture_indices) + (1 in fracture_indices)
    n_end = 2 in fracture_indices
    n_frac = n_int + n_end
    print(gb)
    assert gb.num_graph_nodes() == (1 + 3 * n_frac + (1 - n_end))

    # 3d-2d: n_frac between matrix and fractures,
    # 2d-0d: n_frac
    # 1d-0d: 2 between well and intersection for each internal fracture and 1 for
    # endpoint fracture
    assert gb.num_graph_edges() == (n_frac + n_frac + 2 * n_int + n_end)

    # Only the first well grid should be on the global boundary
    boundary_faces = [[1, 0], [0, 0], [0, 0]]
    for ind, well_grid in enumerate(gb.grids_of_dimension(1)):
        assert np.all(np.isclose(well_grid.tags["fracture_faces"], fracture_faces[ind]))
        assert np.all(np.isclose(well_grid.tags["tip_faces"], tip_faces[ind]))
        assert np.all(
            np.isclose(well_grid.tags["domain_boundary_faces"], boundary_faces[ind])
        )


# Single fracture: internal to well 0 and tip for well 1
# First two well grids (first dimension below) correspond to well 0, the last grid to well 1
f_tags_0 = [[0, 1], [1, 0], [0, 1]]
t_tags_0 = [[0, 0], [0, 1], [0, 0]]
b_tags_0 = [[1, 0], [0, 0], [1, 0]]
# Number of grids and intersections, numbers in sums sorted by descending dimension.
# Grids: One 3d, 1 fracture, 2 + 1 well grids and 1 + 1 intersections.
# Interfaces: 1 3d-2d, 2+1 well-fracture 1+1 fracture-intersection
gb_data_0 = [1 + 1 + 3 + 2, 1 + 3 + 2]

# All three fractures. frac 2 only intersects well 0
# First three well grids (first dimension below) correspond to well 0,
# the last three grids to well 1
f_tags_1 = [[0, 1], [1, 1], [1, 1], [0, 1], [1, 1]]
t_tags_1 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
b_tags_1 = [[1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]
# Number of grids and intersections, numbers in sums sorted by descending dimension.
# Grids: One 3d, 3 fracture, 3 + 2 well grids and 3 + 2 intersections.
# Interfaces: 3 3d-2d, 3+2 well-fracture and 5+3 fracture-intersection
gb_data_1 = [1 + 3 + 5 + 5, 3 + 5 + 8]


@pytest.mark.parametrize(
    "fracture_indices, fracture_faces, tip_faces, boundary_faces, gb_data",
    [
        ([1], f_tags_0, t_tags_0, b_tags_0, gb_data_0),
        ([0, 1, 2], f_tags_1, t_tags_1, b_tags_1, gb_data_1),
    ],
)
def test_add_two_wells(
    fracture_indices: List[int],
    fracture_faces: List[List[int]],
    tip_faces: List[List[int]],
    boundary_faces: List[List[int]],
    gb_data: List[int],
) -> None:
    """Compute intersection between two well and the fracture network, mesh and
    add well grids to gb.

    Parameters:
        fracture_indices (list): which fractures to use.
        fracture_faces (list): Each item is the expected fracture face tags for one
            well grid, assumed to have two faces each.
        tip_faces (list): Each item is the expected tip face tags for one well grid,
            assumed to have two faces each.
        boundary_faces (list): Each item is the expected boundary face tags for one
            well grid, assumed to have two faces each.
        gb_data (list): expected number of grids and number of edges.
    """
    gb = _generate_gb(fracture_indices, [0, 1])
    assert np.isclose(gb.num_graph_nodes(), gb_data[0])
    assert np.isclose(gb.num_graph_edges(), gb_data[1])

    # Only the first well grid should be on the global boundary
    for ind, well_grid in enumerate(gb.grids_of_dimension(1)):
        assert np.all(np.isclose(well_grid.tags["fracture_faces"], fracture_faces[ind]))
        assert np.all(np.isclose(well_grid.tags["tip_faces"], tip_faces[ind]))
        assert np.all(
            np.isclose(well_grid.tags["domain_boundary_faces"], boundary_faces[ind])
        )
