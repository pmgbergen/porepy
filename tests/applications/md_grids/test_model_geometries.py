import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    NonMatchingSquareDomainOrthogonalFractures,
)


class TestModel(NonMatchingSquareDomainOrthogonalFractures, pp.SinglePhaseFlow):
    """Single phase flow with a (potentially) non-matching grid."""


@pytest.mark.parametrize(
    "fracture_indices, fracture_refinement_ratios, interface_refinement_ratio, expected_fracture_cell_num, expected_interface_cell_num",
    [
        ([0, 1], [2, 3], 3, [4, 6], [12, 12]),
        ([0], [5], 4, [10], [16]),
        ([1], [1], 2, [2], [8]),
        ([0, 1], [1, 1], 1, [2, 2], [4, 4]),
    ],
)
def test_nonmatching_grid_generation(
    fracture_indices,
    fracture_refinement_ratios,
    interface_refinement_ratio,
    expected_fracture_cell_num,
    expected_interface_cell_num,
):
    """Testing generation of non-matching grid.

    We here test that the expected geometry is created when generating a non-matching
    mixed-dimensional grid. Specifically we monitor whether the number of fracture and
    interface cells are as expected given the refinement ratios we test for. The grids
    we test are non-matching in the sense that interface-fracture grids are
    non-matching, as well as interface-matrix grids are non-matching.

    The geometry is a 2d unit square with 2x2 cells. Depending on the parameter
    `fracture_indices` (see documentation below), the grid may contain 0, 1 or 2
    fractures.

    Parameters:
        fracture_indices: List of (up to two) fracture indices as found in
            :class:`SquareDomainOrthogonalFractures`.
        fracture_refinement_ratios: The ratio(s) of which the fracture(s) should be
            refined. The fracture corresponding to the n-th element in
            `fracture_indices` will be refined by the ratio corresponding to the n-th
            element in `fracture_refinement_ratios`.
        interface_refinement_ratio: The ratio we want to refine the interface grid with.
        expected_fracture_cell_num: The expected number of fracture cells after
            refinement by the ratio(s) found in `fracture_refinement_ratios`.
        expected_interface_cell_num: The expected number of interface cells
            refinement by the ratio found in `interface_refinement_ratios`.

    """
    params = {
        "fracture_indices": fracture_indices,
        "fracture_refinement_ratios": fracture_refinement_ratios,
        "interface_refinement_ratio": interface_refinement_ratio,
    }

    model = TestModel(params)
    pp.run_time_dependent_model(model, params)

    for i, fracture in enumerate(model.mdg.subdomains(dim=1)):
        assert fracture.num_cells == expected_fracture_cell_num[i]

    for i, interface in enumerate(model.mdg.interfaces(dim=1)):
        assert int(interface.num_cells) == expected_interface_cell_num[i]
