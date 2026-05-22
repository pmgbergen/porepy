"""Testing functionality related to the fracture importer. These functions are covered:
- network_2d_from_csv
- network_3d_from_csv
- elliptic_network_3d_from_csv

"""

from pathlib import Path
from typing import Callable, Generator, Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.applications.test_utils.fracture_properties import (
    distance_from_points_to_fracture_plane,
)
from porepy.fracs import fracture_importer
from porepy.fracs.fracture_network_3d import FractureNetwork3d


@pytest.fixture(params=[2, 3])
def nd(request) -> Literal[2, 3]:
    return request.param


@pytest.fixture
def file_name(tmp_path: Path) -> Path:
    return tmp_path / "fracture_network.csv"


@pytest.fixture
def fractures(nd: Literal[2, 3]) -> list[np.ndarray]:
    if nd == 2:
        # Each line is a fracture defined by its endpoints (x1, y1, x2, y2).
        f_0 = np.array([[0.2, 0.2, 0.8, 0.8]])
        f_1 = np.array([[0.2, 0.8, 0.8, 0.2]])
        f_2 = np.array([[0.5, 0.0, 0.5, 1.0]])

    else:  # nd == 3
        f_0 = np.array(  # Plane fracture with normal in z direction.
            [[0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8], [0.5, 0.5, 0.5, 0.5]]
        )
        # Elliptic fracture.
        f_1 = np.array([[0.5, 0.5, 0.6, 0.4, 0.2, np.pi / 6, np.pi / 3, np.pi / 4]])
        # Plane fracture with normal in y direction, and five points.
        f_2 = np.array(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 1.0, 1.0, 0.5, 0.0],
                [0.2, 0.2, 0.8, 0.9, 0.8],
            ]
        )

    return [f_0, f_1, f_2]


@pytest.fixture
def domain(nd: Literal[2, 3]) -> np.ndarray:
    if nd == 2:
        domain = np.array([[0, 0, 1, 1]])
    else:  # nd == 3
        domain = np.array([[0, 0, 0, 1, 1, 1]])
    return domain


@pytest.mark.parametrize("has_domain", [True, False])
@pytest.mark.parametrize("num_fracs", [0, 1, 2, 3])
@pytest.mark.parametrize("include_comments", [True, False])
def test_fracture_importer(
    nd, has_domain, num_fracs, include_comments: bool, file_name, fractures, domain
):
    """Test importing fracture networks from CSV files in 2D and 3D.

    Parameters:
        nd: Number of spatial dimensions (2 or 3).
        has_domain: Whether to include domain specification in the CSV file.
        num_fracs: Number of fractures to include in the CSV file.
        include_comments: Whether to include comment lines in the CSV file.
        file_name: Path to the temporary CSV file.
        fractures: List of fracture definitions as numpy arrays.
        domain: Domain specification as a numpy array.

    """
    loc_fractures = fractures[:num_fracs]

    with open(file_name, "w") as f:
        if include_comments:
            f.write("# This is a comment line\n")
        if has_domain:
            d = domain.flatten()
            f.write(",".join([str(val) for val in d]) + "\n")
        if include_comments:
            f.write("# Another comment line\n")
        for frac in loc_fractures:
            vals = frac.ravel("F")
            f.write(",".join([str(val) for val in vals]) + "\n")
            if include_comments:
                f.write("# Comment after fracture\n")

    if num_fracs == 0 and not has_domain:
        with pytest.raises(ValueError):
            _ = fracture_importer.network_from_csv(file_name, has_domain=has_domain)
        return

    network = pp.fracture_importer.network_from_csv(file_name, has_domain=has_domain)

    # Verify domain geometry.
    if has_domain:
        for key, val in network.domain.bounding_box.items():
            if key.endswith("min"):
                assert val == 0.0
            else:  # key.endswith("max")
                assert val == 1.0
    else:
        assert network.domain is None

    # Verify number of fractures.
    assert network.num_frac() == num_fracs
    # Verify fracture geometries.
    for fi, f_known in enumerate(loc_fractures):
        if nd == 2 or (nd == 3 and fi != 1):
            f_imported = network.fractures[fi].pts
            assert compare_arrays(f_known.reshape((nd, -1), order="F"), f_imported)
        else:  # nd == 3 and fi == 1 (elliptic fracture)
            # We have to generate points on the elliptic fracture to compare: One is the
            # center, two others will be half the distance from the center in the
            # direction of the major and minor axis, respectively. If the distance
            # between the fracture and each of these points is zero, the fracture must
            # be in the right plane, which is the main property we want to test.
            f_known = f_known.ravel()
            center = f_known[:3].ravel()

            # Length of major and minor axis.
            major_axis = f_known[3]
            minor_axis = f_known[4]
            major_axis_angle = f_known[5]
            # Vectors along the minor and major axis before strike-dip rotation, i.e.,
            # in the xy-plane.
            vec_major_xy = np.array(
                [np.cos(major_axis_angle), np.sin(major_axis_angle), 0]
            )
            vec_minor_xy = np.array(
                [-np.sin(major_axis_angle), np.cos(major_axis_angle), 0]
            )

            strike_angle = f_known[6]
            dip_angle = f_known[7]
            # The strike gives a rotation in the xy-plane:
            rot_strike = np.array(
                [
                    [np.cos(strike_angle), -np.sin(strike_angle), 0],
                    [np.sin(strike_angle), np.cos(strike_angle), 0],
                    [0, 0, 1],
                ]
            )
            # Use Rodrigues' rotation formula (implemented in the called function) to
            # get the rotation matrix for the dip.
            rot_dip = pp.map_geometry.rotation_matrix(
                dip_angle,
                vect=np.array([np.cos(strike_angle), np.sin(strike_angle), 0]),
            )
            # Apply first strike rotation, then dip rotation around the rotated major
            # axis.
            major_axis_rotated = rot_dip @ rot_strike @ vec_major_xy
            minor_axis_rotated = rot_dip @ rot_strike @ vec_minor_xy
            # Control points.
            point_major = center + 0.5 * major_axis * major_axis_rotated
            point_minor = center + 0.5 * minor_axis * minor_axis_rotated

            for point in [center, point_major, point_minor]:
                dis = distance_from_points_to_fracture_plane(
                    point, center, strike_angle, dip_angle
                )
                assert np.abs(dis).max() <= 1e-6
