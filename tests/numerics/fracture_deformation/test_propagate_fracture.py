""" Tests related to fracture propagation.
"""
import numpy as np
import pytest

import porepy as pp

## Below follows methods to test the partial update of FV discretization schemes under
# fracture propagation.


def _two_fractures_overlapping_regions():
    # Region with two fractures that are pretty close; the update stencil for fv methods
    # due to update of each over them will basically include the entire grid.
    # Two prolongation steps for the fracture.
    # In the first step, one fracture grows, one is stuck.
    # In second step, the second fracture grows in both ends

    frac = [np.array([[1, 2], [1, 1]]), np.array([[2, 3], [2, 2]])]
    mdg = pp.meshing.cart_grid(frac, [6, 3])
    split_scheme = [
        [np.array([29]), np.array([])],
        [np.array([30]), np.array([34, 36])],
    ]
    return mdg, split_scheme


def _two_fractures_non_overlapping_regions():
    # Two fractures far away from each other. The update stencils for fv methods will
    # be non-overlapping, meaning the method must deal with non-connected subgrids.
    frac = [np.array([[1, 1], [1, 2]]), np.array([[8, 8], [1, 2]])]
    mdg = pp.meshing.cart_grid(frac, [9, 3])
    split_scheme = [[np.array([1]), np.array([8])]]
    return mdg, split_scheme


def _two_fractures_regions_become_overlapping():
    # Two fractures far away from each other. In the first step, the update stencils for
    # fv methods will be non-overlapping, meaning the method must deal with
    # non-connected subgrids. In the second step, the regions are overlapping.
    frac = [np.array([[1, 1], [1, 2]]), np.array([[8, 8], [1, 2]])]
    mdg = pp.meshing.cart_grid(frac, [9, 3])
    split_scheme = [[np.array([1]), np.array([8])], [np.array([49]), np.array([55])]]
    return mdg, split_scheme


# The main test function for
@pytest.mark.parametrize(
    "geometry",
    [
        _two_fractures_overlapping_regions,
        _two_fractures_non_overlapping_regions,
        _two_fractures_regions_become_overlapping,
    ],
)
@pytest.mark.parametrize(
    "discretization",
    [pp.Mpfa("flow"), pp.Mpsa("mechanics"), pp.Biot("mechanics")],
)
def test_partial_discretization(geometry, discretization):
    """Method to test (FV) partial discretization under fracture propagation.

    The test is based on first discretizing, and then do one or several fracture
    propagation steps. after each step, we do a partial update of the discretization
    scheme, and compare with a full discretization on the newly split grid. The test
    fails unless all discretization matrices generated are identical.

    NOTE: Only the highest-dimensional grid in the MixedDimensionalGrid is used.

    """

    # Get MixedDimensionalGrid and splitting schedule
    mdg, faces_to_split = geometry()

    sd_top = mdg.subdomains(dim=mdg.dim_max())[0]
    g_1, g_2 = mdg.subdomains(dim=1)

    # Make the splitting schedule on the format expected by fracture propagation
    split_faces = []
    for face in faces_to_split:
        split_faces.append({g_1: face[0], g_2: face[1]})

    def set_param(sd, data):
        # Helper method to set parameters.
        # For Mpfa and Mpsa, some of the data is redundant, but that should be fine;
        # it allows us to use a single parameter function
        data[pp.PARAMETERS] = {}
        data[pp.PARAMETERS]["flow"] = {
            "bc": pp.BoundaryCondition(sd),
            "bc_values": np.zeros(sd.num_faces),
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells)),
            "mass_weight": np.ones(sd.num_cells),
        }

        data[pp.PARAMETERS]["mechanics"] = {
            "bc": pp.BoundaryConditionVectorial(sd),
            "bc_values": np.zeros(sd.num_faces * sd.dim),
            "fourth_order_tensor": pp.FourthOrderTensor(
                np.ones(sd.num_cells), np.ones(sd.num_cells)
            ),
            "scalar_vector_mappings": {'foo': 1,
                 'bar': pp.SecondOrderTensor(np.ones(sd.num_cells))},
        }

    # Populate parameters
    data = mdg.subdomain_data(sd_top)
    data[pp.DISCRETIZATION_MATRICES] = {"flow": {}, "mechanics": {}}
    set_param(sd_top, data)
    # Discretize
    discretization.discretize(sd_top, data)

    # Loop over propagation steps
    for split in split_faces:
        # Split the face
        pp.propagate_fracture.propagate_fractures(mdg, split)

        # Make parameters for the new grid
        data = mdg.subdomain_data(sd_top)
        set_param(sd_top, data)

        # Transfer information on new faces and cells from the format used
        # by self.evaluate_propagation to the format needed for update of
        # discretizations (see Discretization.update_discretization()).
        new_faces = data.get("new_faces", np.array([], dtype=int))
        split_faces = data.get("split_faces", np.array([], dtype=int))
        modified_faces = np.hstack((new_faces, split_faces))
        update_info = {
            "map_cells": data["cell_index_map"],
            "map_faces": data["face_index_map"],
            "modified_cells": data.get("new_cells", np.array([], dtype=int)),
            "modified_faces": modified_faces,
        }
        data["update_discretization"] = update_info

        # Update the discretization
        discretization.update_discretization(sd_top, data)

        # Create a new data dictionary, populate
        new_d = {}
        new_d[pp.DISCRETIZATION_MATRICES] = {"flow": {}, "mechanics": {}}
        set_param(sd_top, new_d)

        # Full discretization
        discretization.discretize(sd_top, new_d)

        # Compare discretization matrices
        for eq_type in ["flow", "mechanics"]:  # We know which keywords were used
            for key in data[pp.DISCRETIZATION_MATRICES][eq_type]:
                assert key in new_d[pp.DISCRETIZATION_MATRICES][eq_type]
            for key in new_d[pp.DISCRETIZATION_MATRICES][eq_type]:
                assert key in data[pp.DISCRETIZATION_MATRICES][eq_type]

            for key, mat in data[pp.DISCRETIZATION_MATRICES][eq_type].items():
                if isinstance(mat, dict):
                    # Special treatment of Biot coupling terms
                    for sub_key, sub_mat in mat.items():
                        mat2 = new_d[pp.DISCRETIZATION_MATRICES][eq_type][key][sub_key]
                        assert sub_mat.shape == mat2.shape
                        diff = sub_mat - mat2
                        if diff.data.size > 0:
                            assert np.max(np.abs(diff.data)) < 1e-10
                else:
                    mat2 = new_d[pp.DISCRETIZATION_MATRICES][eq_type][key]
                    assert mat.shape == mat2.shape
                    diff = mat - mat2

                    if diff.data.size > 0:
                        assert np.max(np.abs(diff.data)) < 1e-10
