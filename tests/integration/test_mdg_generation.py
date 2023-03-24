"""

Collection of tests for validating the mixed-dimensional grid generation

Functionalities being tested:
* Generation with/without fractures
* Generation of meshes with type {"simplex", "cartesian", "tensor_grid"}
* Generation of meshes with dimension {2,3}

"""
import unittest
from typing import List, Literal, Optional, Union

import numpy as np
import pytest

import porepy as pp
import porepy.grids.standard_grids.utils as utils


class TestMDGridGeneration:
    """Test suit for verifying the md-grid generation of pp.create_mdg"""

    def h_reference(self) -> float:
        return 0.5

    def fracture_2d_data(self) -> List[np.array]:
        data: List[np.array] = [
            np.array([[0.0, 2.0], [0.0, 0.0]]),
            np.array([[1.0, 1.0], [0.0, 1.0]]),
            np.array([[2.0, 2.0], [0.0, 2.0]]),
        ]
        return data

    def fracture_3d_data(self) -> List[np.array]:
        data: List[np.array] = [
            np.array(
                [[2.0, 3.0, 3.0, 2.0], [2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]
            ),
            np.array(
                [[2.0, 3.0, 3.0, 2.0], [1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            np.array(
                [[1.0, 4.0, 4.0, 1.0], [3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 2.0, 2.0]]
            ),
        ]
        return data

    def mdg_types(self):
        return ["simplex", "cartesian", "tensor_grid"]

    def domains(self):
        domain_2d = pp.Domain({"xmin": 0, "xmax": 5, "ymin": 0, "ymax": 5})
        domain_3d = pp.Domain(
            {"xmin": 0, "xmax": 5, "ymin": 0, "ymax": 5, "zmin": 0, "zmax": 5}
        )
        return [domain_2d, domain_3d]

    # Extra mesh arguments
    def extra_args_data_2d(self) -> List[dict[str]]:
        simplex_extra_args: dict[str] = {"mesh_size_bound": 1.0, "mesh_size_min": 0.1}
        cartesian_extra_args: dict[str] = {"nx": [10, 10], "physdims": [5, 5]}
        tensor_grid_extra_args: dict[str] = {
            "x": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "y": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        }
        return [simplex_extra_args, cartesian_extra_args, tensor_grid_extra_args]

    def extra_args_data_3d(self) -> List[dict[str]]:
        simplex_extra_args: dict[str] = {"mesh_size_bound": 1.0, "mesh_size_min": 0.1}
        cartesian_extra_args: dict[str] = {"nx": [10, 10, 10], "physdims": [5, 5, 5]}
        tensor_grid_extra_args: dict[str] = {
            "x": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "y": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "z": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        }

        return [simplex_extra_args, cartesian_extra_args, tensor_grid_extra_args]

    def generate_network(self, domain_index, fracture_indices: List[int]):
        """Construct fracture network.

        Parameters:
            domain_index (int): index of computational domain
            fracture_indices (list): combination of fractures

        Returns:
            Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: Collection of Fractures with
            geometrical information
        """

        disjoint_fractures = None
        domain: pp.Domain = self.domains()[domain_index]
        if domain.dim == 2:
            geometry = [self.fracture_2d_data()[id] for id in fracture_indices]
            disjoint_fractures = list(map(pp.LineFracture, geometry))
        elif domain.dim == 3:
            geometry = [self.fracture_3d_data()[id] for id in fracture_indices]
            disjoint_fractures = list(map(pp.PlaneFracture, geometry))

        network = pp.create_fracture_network(disjoint_fractures, domain)
        return network

    def high_level_mdg_generation(self, grid_type, fracture_network):
        """Generates a mixed-dimensional grid.

        Parameters:
            fracture_network Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: selected
            pp.FractureNetwork<n>d with n in {2,3}

        Returns:
            pp.MixedDimensionalGrid: Container of grids and its topological relationship
             along with a surrounding matrix defined by domain.
        """
        # common mesh argument
        mesh_arguments: dict[str] = {"mesh_size": self.h_reference()}

        extra_arg_index = self.mdg_types().index(grid_type)
        extra_arguments = None
        if fracture_network.domain.dim == 2:
            extra_arguments = self.extra_args_data_2d()[extra_arg_index]
        elif fracture_network.domain.dim == 3:
            extra_arguments = self.extra_args_data_3d()[extra_arg_index]
        mdg = pp.create_mdg(
            grid_type, mesh_arguments, fracture_network, **extra_arguments
        )
        return mdg

    def low_level_mdg_generation(self, grid_type, fracture_network):
        """Generates a mixed-dimensional grid.

        Parameters:
            fracture_network Union[pp.FractureNetwork2d, pp.FractureNetwork3d]: selected
            pp.FractureNetwork<n>d with n in {2,3}

        Returns:
            pp.MixedDimensionalGrid: Container of grids and its topological relationship
             along with a surrounding matrix defined by domain.
        """

        lower_level_arguments = None
        if fracture_network.domain.dim == 2:
            lower_level_arguments = self.extra_args_data_2d()[
                self.mdg_types().index(grid_type)
            ]
        elif fracture_network.domain.dim == 3:
            lower_level_arguments = self.extra_args_data_3d()[
                self.mdg_types().index(grid_type)
            ]

        if grid_type == "simplex":
            lower_level_arguments["mesh_size_frac"] = self.h_reference()
            utils.set_mesh_sizes(lower_level_arguments)
            mdg = fracture_network.mesh(lower_level_arguments)
            return mdg

        elif grid_type == "cartesian":

            n_cells = lower_level_arguments["nx"]
            phys_dims = lower_level_arguments["physdims"]

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.cart_grid(fracs=fractures, **lower_level_arguments)
            return mdg

        elif grid_type == "tensor_grid":
            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.tensor_grid(fracs=fractures, **lower_level_arguments)
            return mdg

    def mdg_equality(self, h_mdg, l_mdg) -> bool:
        """Relaxed mdg equality"""
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
        """Test the generated mdg object."""

        fracture_network = self.generate_network(domain_index, fracture_indices)
        h_mdg = self.high_level_mdg_generation(grid_type, fracture_network)
        l_mdg = self.low_level_mdg_generation(grid_type, fracture_network)
        equality_q = self.mdg_equality(h_mdg, l_mdg)
        assert equality_q


class TestGenerationInconsistencies(TestMDGridGeneration):
    """Test suit for verifying function inconsistencies."""

    def test_grid_type_inconsistencies(self):

        fracture_network = self.generate_network(0, [0, 1, 2])
        mesh_arguments: dict[str] = {"mesh_size": self.h_reference()}

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

    def test_mesh_arguments_inconsistencies(self):

        grid_type = "simplex"
        fracture_network = self.generate_network(0, [0, 1, 2])

        with pytest.raises(TypeError) as error_message:
            mesh_arguments = [self.h_reference()]
            ref_msg = str(
                "mesh_arguments must be dict[str], not %r" % type(mesh_arguments)
            )
            pp.create_mdg(grid_type, mesh_arguments, fracture_network)
        assert ref_msg in str(error_message.value)

        with pytest.raises(TypeError) as error_message:
            mesh_size = complex(1, 2)
            mesh_arguments: dict[str] = {"mesh_size": mesh_size}
            ref_msg = str("mesh_size must be float, not %r" % type(mesh_size))
            pp.create_mdg(grid_type, mesh_arguments, fracture_network)
        assert ref_msg in str(error_message.value)

        with pytest.raises(ValueError) as error_message:
            mesh_size = -1.0
            mesh_arguments: dict[str] = {"mesh_size": mesh_size}
            ref_msg = str("mesh_size must be strictly positive %r" % mesh_size)
            pp.create_mdg(grid_type, mesh_arguments, fracture_network)
        assert ref_msg in str(error_message.value)

    def test_network_inconsistencies(self):

        grid_type = "simplex"
        fracture_network = self.generate_network(0, [0, 1, 2])
        mesh_arguments: dict[str] = {"mesh_size": self.h_reference()}

        with pytest.raises(ValueError) as error_message:
            fracture_network.domain.dim = 0
            ref_msg = str(
                "Inferred dimension must be 2 or 3, not %r"
                % fracture_network.domain.dim
            )
            pp.create_mdg(grid_type, mesh_arguments, fracture_network)
            fracture_network.domain.dim = 2
        assert ref_msg in str(error_message.value)

        with pytest.raises(TypeError) as error_message:
            ref_msg = str(
                "fracture_network must be FractureNetwork2d or FractureNetwork3d, not %r"
                % type(complex(1, 2))
            )
            pp.create_mdg(grid_type, mesh_arguments, complex(1, 2))
        assert ref_msg in str(error_message.value)

