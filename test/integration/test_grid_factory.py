"""
"""
import unittest

import numpy as np

import porepy as pp


class TestSimpleMeshing(unittest.TestCase):
    def test_nested_simple(self):
        # Simple test of the nested generation. Both 2d and 3d domains.
        # Main check: The refinement is indeed by splitting
        mesh_args = {"mesh_size_frac": 1, "mesh_size_bound": 1}

        num_ref = 3

        params = {
            "mode": "nested",
            "num_refinements": num_ref,
            "mesh_param": mesh_args,
        }

        network_2d = pp.FractureNetwork2d(
            domain={"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        )
        network_3d = pp.FractureNetwork2d(
            domain={"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        )
        for network in [network_2d, network_3d]:
            factory = pp.refinement.GridSequenceFactory(network, params)

            num_cells_prev = 0

            for counter, gb in enumerate(factory):
                assert counter < num_ref
                dim = gb.dim_max()
                if counter > 0:
                    assert gb.num_cells() == num_cells_prev * (2 ** dim)
                num_cells_prev = gb.num_cells()

    def test_unstructured(self):
        # Use unstructured meshing, no relation between grids on different levels
        mesh_args = [
            {"mesh_size_frac": 1, "mesh_size_bound": 1},
            {"mesh_size_frac": 0.5, "mesh_size_bound": 0.5},
        ]

        num_ref = len(mesh_args)

        params = {
            "mode": "unstructured",
            "num_refinements": num_ref,
            "mesh_param": mesh_args,
        }

        # Add a single fracture to the network
        pts = np.array([[0.3, 0.7], [0.5, 0.5]])
        edges = np.array([[0], [1]])
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(pts, edges, domain)

        factory = pp.refinement.GridSequenceFactory(network, params)

        for gb in factory:
            # It is not really clear what we can test here.
            pass

    def test_nested_pass_grid_args(self):
        # Check that grid arguments to the fracture network meshing ('grid_param' below)
        # are correctly passed
        mesh_args = {"mesh_size_frac": 1, "mesh_size_bound": 1}

        num_ref = 2

        # Add a single fracture to the network
        pts = np.array([[0.3, 0.7], [0.5, 0.5]])
        edges = np.array([[0], [1]])
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(pts, edges, domain)

        params = {
            "mode": "nested",
            "num_refinements": num_ref,
            "mesh_param": mesh_args,
            # The fracture is a constraint
            "grid_param": {"constraints": np.array([0])},
        }

        factory = pp.refinement.GridSequenceFactory(network, params)

        for counter, gb in enumerate(factory):
            dim = gb.dim_max()
            for dim in range(dim):
                # Since there are no true fractures in the network (only constraints)
                # there should be no lower-dimensional grids
                assert len(gb.grids_of_dimension(dim)) == 0


if __name__ == "__main__":
    TestSimpleMeshing().test_unstructured()
    unittest.main()
