"""

Collection of tests for validating the mixed-dimensional grid generation

Functionalities being tested:
* Generation with/without fractures
* Generation of meshes with type {"simplex", "cartesian", "tensor_grid"}
* Generation of meshes with dimension {2,3}
* Generation of simplicial DFN meshes with dimension {2, 3}

"""
from __future__ import annotations

from typing import List, Union

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

FractureNetwork = Union[FractureNetwork2d, FractureNetwork3d]


class TestMDGridGeneration:
    """Test suite for verifying the md-grid generation of pp.create_mdg"""

    def cell_size(self) -> float:
        """Common cell_size for all tests"""
        return 0.5

    def fracture_2d_data(self) -> List[np.ndarray]:
        """Fracture points for 2d cases"""
        data: List[np.array] = [
            # Fractures contained within the domain
            np.array([[0.0, 2.0], [1.0, 1.0]]),
            np.array([[1.0, 1.0], [0.0, 1.0]]),
            np.array([[2.0, 2.0], [0.0, 2.0]]),
            # A fracture outside the domain
            np.array([[-1.0, -1.0], [-1.0, -2.0]]),
        ]
        return data

    def fracture_3d_data(self) -> List[np.array]:
        """Fracture points for 3d cases"""
        data: List[np.array] = [
            # Fractures inside the domain
            np.array(
                [[2.0, 3.0, 3.0, 2.0], [2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]
            ),
            np.array(
                [[2.0, 3.0, 3.0, 2.0], [1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            np.array(
                [[1.0, 4.0, 4.0, 1.0], [3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 2.0, 2.0]]
            ),
            # One fracture outside the domain (note minus sign in front)
            -np.array(
                [[2.0, 3.0, 3.0, 2.0], [2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]
            ),
        ]
        return data

    def mdg_types(self):
        """Supported mdg types"""
        return ["simplex", "cartesian", "tensor_grid"]

    def domains(self):
        """Domain list"""
        domain_2d = pp.Domain({"xmax": 5, "ymax": 5})
        domain_3d = pp.Domain({"xmax": 5, "ymax": 5, "zmax": 5})
        return [domain_2d, domain_3d]

    # Extra mesh arguments
    def higher_level_extra_args_data_2d(self) -> List[dict]:
        """Admissible keys in pp.create_mdg for 2d cases"""
        simplex_extra_args: dict = {
            "cell_size_min": 0.5,
            "cell_size_boundary": 1.0,
            "cell_size_fracture": 0.5,
        }
        cartesian_extra_args: dict = {"cell_size_x": 0.5, "cell_size_y": 0.5}
        tensor_grid_extra_args: dict = {
            "x_pts": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "y_pts": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        }
        return [simplex_extra_args, cartesian_extra_args, tensor_grid_extra_args]

    def lower_level_extra_args_data_2d(self) -> List[dict]:
        """Admissible keys for 2d cases"""
        simplex_extra_args: dict = {
            "mesh_size_min": 0.5,
            "mesh_size_bound": 1.0,
            "mesh_size_frac": 0.5,
        }
        cartesian_extra_args: dict = {"nx": [10, 10], "physdims": [5, 5]}
        tensor_grid_extra_args: dict = {
            "x": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "y": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        }
        return [simplex_extra_args, cartesian_extra_args, tensor_grid_extra_args]

    def higher_level_extra_args_data_3d(self) -> List[dict]:
        """Admissible keys in pp.create_mdg for 3d cases"""
        simplex_extra_args: dict = {
            "cell_size_min": 0.5,
            "cell_size_boundary": 1.0,
            "cell_size_fracture": 0.5,
        }
        cartesian_extra_args: dict = {
            "cell_size_x": 0.5,
            "cell_size_y": 0.5,
            "cell_size_z": 0.5,
        }
        tensor_grid_extra_args: dict = {
            "x_pts": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "y_pts": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "z_pts": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        }
        return [simplex_extra_args, cartesian_extra_args, tensor_grid_extra_args]

    def lower_level_extra_args_data_3d(self) -> List[dict]:
        """Admissible keys for 3d cases"""
        simplex_extra_args: dict = {
            "mesh_size_min": 0.5,
            "mesh_size_bound": 1.0,
            "mesh_size_frac": 0.5,
        }
        cartesian_extra_args: dict = {"nx": [10, 10, 10], "physdims": [5, 5, 5]}
        tensor_grid_extra_args: dict = {
            "x": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "y": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "z": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        }

        return [simplex_extra_args, cartesian_extra_args, tensor_grid_extra_args]

    def generate_network(
        self, domain_index: int, fracture_indices: List[int]
    ) -> FractureNetwork:
        """Construct fracture network:
            - FractureNetwork2d
            - FractureNetwork3d

        Parameters:
            domain_index (int): index of computational domain
            fracture_indices (list): combination of fractures

        Returns:
            Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: Collection of Fractures with
            geometrical information
        """

        disjoint_fractures: list = None
        domain: pp.Domain = self.domains()[domain_index]
        if domain.dim == 2:
            # Collect fracture points (the geometry of each fracture)
            geometry = [self.fracture_2d_data()[id] for id in fracture_indices]
            # Build a disjoint set of LineFractures
            disjoint_fractures = list(map(pp.LineFracture, geometry))
        elif domain.dim == 3:
            # Collect fracture points (the geometry of each fracture)
            geometry = [self.fracture_3d_data()[id] for id in fracture_indices]
            # Build a disjoint set of PlaneFracture
            disjoint_fractures = list(map(pp.PlaneFracture, geometry))

        network = pp.create_fracture_network(disjoint_fractures, domain)
        return network, domain

    def high_level_mdg_generation(self, grid_type, fracture_network):
        """Generates a mixed-dimensional grid using pp.create_mdg

        Parameters:
            fracture_network Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: selected
            pp.FractureNetwork<n>d with n in {2,3}

        Returns:
            pp.MixedDimensionalGrid: Container of grids and its topological relationship
            along with a surrounding matrix defined by domain.
        """
        # common mesh argument
        meshing_args: dict = {"cell_size": self.cell_size()}

        # Collect extra arguments for the test
        extra_arg_index: int = self.mdg_types().index(grid_type)
        extra_arguments: Union[dict, None] = None
        if fracture_network.domain.dim == 2:
            extra_arguments = self.higher_level_extra_args_data_2d()[extra_arg_index]
        elif fracture_network.domain.dim == 3:
            extra_arguments = self.higher_level_extra_args_data_3d()[extra_arg_index]
        meshing_args.update(extra_arguments.items())

        # call high level function
        mdg = pp.create_mdg(
            grid_type, meshing_args, fracture_network, **extra_arguments
        )
        return mdg

    def low_level_mdg_generation(self, grid_type, fracture_network):
        """Generates a mixed-dimensional grid using lower level functions
            - fracture_network.mesh
            - pp.meshing.cart_grid
            - pp.meshing.tensor_grid

        Parameters:
            fracture_network Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: selected
            pp.FractureNetwork<n>d with n in {2,3}

        Returns:
            pp.MixedDimensionalGrid: Container of grids and its topological relationship
            along with a surrounding matrix defined by domain.
        """

        lower_level_arguments = None
        if fracture_network.domain.dim == 2:
            lower_level_arguments = self.lower_level_extra_args_data_2d()[
                self.mdg_types().index(grid_type)
            ]
        elif fracture_network.domain.dim == 3:
            lower_level_arguments = self.lower_level_extra_args_data_3d()[
                self.mdg_types().index(grid_type)
            ]

        if grid_type == "simplex":
            lower_level_arguments["mesh_size_frac"] = self.cell_size()
            lower_level_arguments["mesh_size_bound"] = 2 * self.cell_size()
            lower_level_arguments["mesh_size_min"] = 1 / 5 * self.cell_size()
            mdg = fracture_network.mesh(lower_level_arguments)
            return mdg

        elif grid_type == "cartesian":
            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.cart_grid(fracs=fractures, **lower_level_arguments)
            return mdg

        elif grid_type == "tensor_grid":
            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.tensor_grid(fracs=fractures, **lower_level_arguments)
            return mdg

    def mdg_equality(self, h_mdg, l_mdg) -> bool:
        """Relaxed mdg equality"""

        # This comparison includes information about dimension, involved grids and their
        # topological information
        equality_q = h_mdg.__repr__() == l_mdg.__repr__()
        return equality_q

    test_parameters = [
        ("simplex", 0, []),
        ("simplex", 1, []),
        ("simplex", 0, [0, 1, 2]),
        ("simplex", 1, [0, 1, 2]),
        ("cartesian", 0, []),
        ("cartesian", 1, []),
        ("cartesian", 0, [0, 1, 2]),
        ("cartesian", 1, [0, 1, 2]),
        ("tensor_grid", 0, []),
        ("tensor_grid", 1, []),
        ("tensor_grid", 0, [0, 1, 2]),
        ("tensor_grid", 1, [0, 1, 2]),
    ]

    @pytest.mark.parametrize(
        "grid_type, domain_index, fracture_indices", test_parameters
    )
    def test_generation(self, grid_type, domain_index, fracture_indices) -> None:
        """Test logic compares a mdg generated using pp.create_mdg with one not generated
        with pp.create_mdg.
        The function pp.create_mdg encapsulates the generation of an mdg by
        utilizing the input provided by the user and calling internal functions
        `fracture_network.mesh`, `pp.meshing.cart_grid` or `pp.meshing.tensor_grid`.
        So, in this test:
        - high-level generation means the actual use of the function pp.create_mdg;
        - lower-level generation means the generation of an mdg for each grid_type
            by calling `fracture_network.mesh`, `pp.meshing.cart_grid` or
            `pp.meshing.tensor_grid`."""

        # Generates a fracture_network that can be without fractures
        fracture_network, _ = self.generate_network(domain_index, fracture_indices)
        h_mdg = self.high_level_mdg_generation(grid_type, fracture_network)

        # Generate the fracture once more to avoid issues relating to processing the
        # geometry of a fracture network twice.
        fracture_network, _ = self.generate_network(domain_index, fracture_indices)
        l_mdg = self.low_level_mdg_generation(grid_type, fracture_network)

        # Failing in equality could mean that:
        # - lower level signatures were changed without updates on the `create_mdg`;
        # - `create_mdg` were updated without mapping signatures to the lower level.
        equality_q = self.mdg_equality(h_mdg, l_mdg)
        assert equality_q

    # Test both 2d and 3d domains. Let the fracture outside the domain be first and last
    # in the list, to cover any special cases under deletion of fractures.
    test_parameters_impose_boundary = [
        (0, [2, 3]),
        (0, [3, 2]),
        (1, [0, 3]),
        (1, [3, 0]),
    ]

    @pytest.mark.parametrize(
        "domain_index, fracture_indices", test_parameters_impose_boundary
    )
    def test_fractures_outside_structured_domain(self, domain_index, fracture_indices):
        """Test that structured meshing of networks with fractures outside the domain
        works.

        Both 2d and 3d networks are tested. Failure is most likely to be caused by
        errors in, or changes to, the functionality to delete fractures in fracture
        networks while imposing external boundaries.

        """
        # The target functionality is used for both Cartesian and tensor grids, but it
        # is common for the two, so it should suffice to check the Cartesian case.
        grid_type = "cartesian"
        fracture_network, domain = self.generate_network(domain_index, fracture_indices)
        # Generate the mesh via the high-level function.
        mdg = self.high_level_mdg_generation(grid_type, fracture_network)

        # The fracture network is set up to have two grids, but one of them is outside
        # the domain, so by now, there should be one fracture (thus two grids) left.
        assert len(mdg.subdomains()) == 2
        # Also verify that all the nodes in the fracture grid is inside the domain, that
        # is, we did not purge the wrong grid.
        frac_domain = mdg.subdomains()[-1]
        for pi in range(frac_domain.num_nodes):
            assert frac_domain.nodes[: domain.dim, pi] in domain


class TestGenerationInconsistencies(TestMDGridGeneration):
    """Test suite for verifying function inconsistencies.
    Each TypeError and ValueError messages are being tested.
    """

    def test_grid_type_inconsistencies(self):
        fracture_network, _ = self.generate_network(0, [0, 1, 2])
        mesh_arguments: dict = {"cell_size": self.cell_size()}

        with pytest.raises(TypeError) as error_message:
            grid_type = complex(1, 2)
            ref_msg = str("grid_type must be str, not %r" % type(grid_type))
            pp.create_mdg(grid_type, mesh_arguments, fracture_network)
        assert ref_msg in str(error_message.value)

        with pytest.raises(ValueError) as error_message:
            grid_type = "Simplex"
            ref_msg = str(
                "grid_type must be in ['simplex', 'cartesian', 'tensor_grid'] not %r"
                % grid_type
            )
            pp.create_mdg(grid_type, mesh_arguments, fracture_network)
        assert ref_msg in str(error_message.value)

    def test_simplex_meshing_args_inconsistencies(self):
        grid_type = "simplex"
        fracture_network, _ = self.generate_network(0, [0, 1, 2])

        # testing meshing_args type
        with pytest.raises(TypeError) as error_message:
            meshing_args = [self.cell_size()]
            ref_msg = str("meshing_args must be dict, not %r" % type(meshing_args))
            pp.create_mdg(grid_type, meshing_args, fracture_network)
        assert ref_msg in str(error_message.value)

        # testing incompleteness in cell_sizes
        cell_size_args = ["cell_size_min", "cell_size_boundary", "cell_size_fracture"]
        meshing_args: dict = {
            "cell_size_min": 0.1,
            "cell_size_boundary": 0.1,
            "cell_size_fracture": 0.1,
        }
        for chunk in cell_size_args:
            loc_meshing_args = {}
            loc_meshing_args.update(meshing_args.items())
            loc_meshing_args.pop(chunk)
            with pytest.raises(ValueError) as error_message:
                ref_msg = str("cell_size or " + chunk + " must be provided.")
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

        # testing types in cell_sizes
        cell_size_args = [
            "cell_size",
            "cell_size_min",
            "cell_size_boundary",
            "cell_size_fracture",
        ]
        meshing_args: dict = {
            "cell_size": 0.1,
            "cell_size_min": 0.1,
            "cell_size_boundary": 0.1,
            "cell_size_fracture": 0.1,
        }
        for chunk in cell_size_args:
            with pytest.raises(TypeError) as error_message:
                loc_meshing_args = {}
                loc_meshing_args.update(meshing_args.items())
                loc_meshing_args[chunk] = complex(1, 2)
                ref_msg = str(chunk + " must be float, not %r" % type(complex(1, 2)))
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

        # testing value error in cell_sizes
        for chunk in cell_size_args:
            with pytest.raises(ValueError) as error_message:
                loc_meshing_args = {}
                loc_meshing_args.update(meshing_args.items())
                loc_meshing_args[chunk] = -1.0
                ref_msg = str(chunk + " must be strictly positive %r" % -1.0)
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

        # testing value error for dfn cases
        grid_type = "cartesian"
        fracture_network.domain = None
        with pytest.raises(ValueError) as error_message:
            cell_size = 0.1
            meshing_args: dict = {"cell_size": cell_size}
            ref_msg = str(
                "fracture_network without a domain is only supported for unstructured "
                "simplex meshes, not for %r" % grid_type
            )
            pp.create_mdg(grid_type, meshing_args, fracture_network)
        assert ref_msg in str(error_message.value)

    def test_cartesian_meshing_args_inconsistencies(self):
        grid_type = "cartesian"

        # testing incompleteness in cell_sizes
        cell_size_args = ["cell_size_x", "cell_size_y", "cell_size_z"]
        meshing_args: dict = {
            "cell_size_x": 0.1,
            "cell_size_y": 0.1,
            "cell_size_z": 0.1,
        }
        for chunk in cell_size_args:
            loc_meshing_args = {}
            loc_meshing_args.update(meshing_args.items())
            loc_meshing_args.pop(chunk)
            with pytest.raises(ValueError) as error_message:
                ref_msg = str("cell_size or " + chunk + " must be provided.")
                fracture_network, _ = self.generate_network(1, [0, 1, 2])
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

        # testing types in cell_sizes
        cell_size_args = ["cell_size", "cell_size_x", "cell_size_y", "cell_size_z"]
        meshing_args: dict = {
            "cell_size": 0.1,
            "cell_size_x": 0.1,
            "cell_size_y": 0.1,
            "cell_size_z": 0.1,
        }
        for chunk in cell_size_args:
            with pytest.raises(TypeError) as error_message:
                loc_meshing_args = {}
                loc_meshing_args.update(meshing_args.items())
                loc_meshing_args[chunk] = complex(1, 2)
                ref_msg = str(chunk + " must be float, not %r" % type(complex(1, 2)))
                fracture_network, _ = self.generate_network(1, [0, 1, 2])
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

        # testing value error in cell_sizes
        for chunk in cell_size_args:
            with pytest.raises(ValueError) as error_message:
                loc_meshing_args = {}
                loc_meshing_args.update(meshing_args.items())
                loc_meshing_args[chunk] = -1.0
                ref_msg = str(chunk + " must be strictly positive %r" % -1.0)
                fracture_network, _ = self.generate_network(1, [0, 1, 2])
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

    def test_tensor_grid_meshing_args_inconsistencies(self):
        grid_type = "tensor_grid"
        fracture_network, _ = self.generate_network(1, [0, 1, 2])

        # # testing incompleteness
        cell_size_args = ["x_pts", "y_pts", "z_pts"]
        pts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        meshing_args: dict = {"x_pts": pts, "y_pts": pts, "z_pts": pts}
        for chunk in cell_size_args:
            loc_meshing_args = {}
            loc_meshing_args.update(meshing_args.items())
            loc_meshing_args.pop(chunk)
            with pytest.raises(ValueError) as error_message:
                ref_msg = str("cell_size or " + chunk + " must be provided.")
                fracture_network, _ = self.generate_network(1, [0, 1, 2])
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

        # testing types of x_pts, y_pts and z_pts
        for chunk in cell_size_args:
            with pytest.raises(TypeError) as error_message:
                loc_meshing_args = {}
                loc_meshing_args.update(meshing_args.items())
                loc_meshing_args[chunk] = complex(1, 2)
                ref_msg = str(
                    chunk + " must be np.ndarray, not %r" % type(complex(1, 2))
                )
                fracture_network, _ = self.generate_network(1, [0, 1, 2])
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

        # testing value error in cell_sizes
        for chunk in cell_size_args:
            with pytest.raises(ValueError) as error_message:
                loc_meshing_args = {}
                loc_meshing_args.update(meshing_args.items())
                pts = np.array([-1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.1])
                loc_meshing_args[chunk] = pts
                ref_msg = str(
                    "The points np.min("
                    + chunk
                    + "), "
                    + "np.max("
                    + chunk
                    + ")"
                    + " must be on the boundary."
                )
                fracture_network, _ = self.generate_network(1, [0, 1, 2])
                pp.create_mdg(grid_type, loc_meshing_args, fracture_network)
            assert ref_msg in str(error_message.value)

    def test_network_inconsistencies(self):
        grid_type = "simplex"
        fracture_network, _ = self.generate_network(0, [0, 1, 2])
        mesh_arguments: dict = {"cell_size": self.cell_size()}

        with pytest.raises(TypeError) as error_message:
            ref_msg = str(
                "fracture_network must be FractureNetwork2d or FractureNetwork3d, not %r"
                % type(complex(1, 2))
            )
            pp.create_mdg(grid_type, mesh_arguments, complex(1, 2))
        assert ref_msg in str(error_message.value)


class TestMDGridGenerationWithoutDomains:
    """Tests whether mdgs are correctly created in the absence of domains.

    The known usage for such cases is meshing of DFNs.

    """

    @pytest.fixture(scope="class")
    def line_fractures(self) -> list[list[pp.LineFracture]]:
        f1 = pp.LineFracture(np.array([[0.5, 0.5], [0.25, 0.75]]))
        f2 = pp.LineFracture(np.array([[0.25, 0.75], [0.5, 0.5]]))
        return [[f1], [f1, f2]]

    @pytest.fixture(scope="class")
    def plane_fractures(self) -> list[list[pp.PlaneFracture]]:
        f1 = pp.PlaneFracture(np.array([[0, 0, 0, 0], [-2, 2, 2, -2], [-2, -2, 2, 2]]))
        f2 = pp.PlaneFracture(np.array([[-2, -2, 2, 2], [0, 0, 0, 0], [-2, 2, 2, -2]]))
        return [[f1], [f1, f2]]

    @pytest.mark.parametrize("fracs_idx", [0, 1])
    def test_mdg_no_domain_in_2d(
        self,
        fracs_idx: int,
        line_fractures: list[list[pp.LineFracture]],
    ) -> None:
        """Check if a mdg is correctly created when domain=None and dfn=True in 2d.

        Parameters:
            fracs_idx: Index to access the items of ``line_fractures``.
            line_fractures: List (of length 2) of lists of line fractures. The first
                item of the list corresponds to a list containing a single vertical
                fracture. The second item of the list corresponds to a list
                containing two intersecting line fractures.

        """
        fn = pp.create_fracture_network(fractures=line_fractures[fracs_idx])
        mdg = pp.create_mdg(
            grid_type="simplex",
            meshing_args={"cell_size": 0.125},
            fracture_network=fn,
            **{"dfn": True},
        )
        if fracs_idx == 0:  # the case of a single line fracture
            assert mdg.dim_max() == 1
            assert mdg.dim_min() == 1
            assert mdg.num_subdomains() == 1
            assert mdg.num_interfaces() == 0
        else:  # the case of two intersecting line fractures
            assert mdg.dim_max() == 1
            assert mdg.dim_min() == 0
            assert mdg.num_subdomains() == 3
            assert mdg.num_interfaces() == 2

    @pytest.mark.parametrize("fracs_idx", [0, 1])
    def test_mdg_no_domain_in_3d(
        self,
        fracs_idx: int,
        plane_fractures: list[list[pp.PlaneFracture]],
    ) -> None:
        """Check if a mdg is correctly created when domain=None and dfn=True in 3d.

        Parameters:
            fracs_idx: Index to access the items of ``plane_fractures``.
            plane_fractures: List (of length 2) of lists of plane fractures. The first
                item of the list corresponds to a list containing a single vertical
                plane fracture. The second item of the list corresponds to a list
                containing two intersecting plane fractures.

        """
        fn = pp.create_fracture_network(fractures=plane_fractures[fracs_idx])
        fn.impose_external_boundary()
        mdg = pp.create_mdg(
            grid_type="simplex",
            meshing_args={"cell_size": 2.0},
            fracture_network=fn,
            **{"dfn": True},
        )
        if fracs_idx == 0:  # the case of a single plane fracture
            assert mdg.dim_max() == 2
            assert mdg.dim_min() == 2
            assert mdg.num_subdomains() == 1
            assert mdg.num_interfaces() == 0
        else:  # the case of two intersecting plane fractures
            assert mdg.dim_max() == 2
            assert mdg.dim_min() == 1
            assert mdg.num_subdomains() == 3
            assert mdg.num_interfaces() == 2
