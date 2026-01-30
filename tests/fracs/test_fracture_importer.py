"""Testing functionality related to the fracture importer. These functions are covered:
- network_2d_from_csv
- network_3d_from_csv
- elliptic_network_3d_from_csv

Created on Wed Dec 12 09:05:31 2018

@author: eke001
"""

from pathlib import Path
from typing import Callable, Generator, Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs import fracture_importer
from porepy.fracs.fracture_network_3d import FractureNetwork3d

# ---------- Testing network_2d_from_csv ----------


@pytest.fixture
def file_name(tmp_path) -> Generator[Path, None, None]:
    file_name = tmp_path / "frac.csv"
    yield file_name


@pytest.fixture(params=[2, 3])
def nd(request) -> Literal[2, 3]:
    return request.param


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
        f_1 = np.array([[0.5, 0.5, 0.6, 0.4, 0.2, np.pi / 3, np.pi / 4, 16]])
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
            _ = fracture_importer.network_from_csv_ek(file_name, has_domain=has_domain)
        return

    network = pp.fracture_importer.network_from_csv_ek(file_name, has_domain=has_domain)

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
            # TODO: When the meshing rework is done, we will have a new implementation
            # of the elliptic fracture, and, as part of the reworked test suite,
            # functionality to verify that an elliptic fracture has the correct
            # geometry.
            pass


def test_single_fracture_2d(file_name):
    p = np.array([0, 0, 1, 1])
    f = np.hstack((0, p))
    np.savetxt(file_name, f, delimiter=",")

    network = fracture_importer.network_from_csv(file_name)
    known_pts = np.array([[0, 1], [0, 1]])
    assert compare_arrays(known_pts, network._pts)
    known_edges = np.array([[0], [1]])
    assert compare_arrays(known_edges, network._edges)
    assert network.domain.bounding_box["xmin"] == 0
    assert network.domain.bounding_box["ymin"] == 0
    assert network.domain.bounding_box["xmax"] == 1
    assert network.domain.bounding_box["ymax"] == 1


def test_return_frac_id(file_name):
    p = np.array([0, 0, 1, 1])
    frac_id = np.random.randint(0, 10)
    f = np.hstack((frac_id, p))
    np.savetxt(file_name, f, delimiter=",")

    network, fid = fracture_importer._network_2d_from_csv(
        file_name, return_frac_id=True
    )

    assert fid.size == 1
    assert fid[0] == frac_id


def test_no_data(file_name):
    np.savetxt(file_name, [], delimiter=",")
    network = fracture_importer._network_2d_from_csv(file_name)
    assert network._pts.shape == (2, 0)
    assert network._edges.shape == (2, 0)
    assert network.domain is None
    assert network.num_frac() == 0


def test_max_num_fracs_keyword(file_name):
    p = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    f = np.hstack((np.arange(2).reshape((-1, 1)), p))
    np.savetxt(file_name, f, delimiter=",")

    # First load one fracture only
    network = fracture_importer._network_2d_from_csv(file_name, max_num_fracs=1)
    known_pts = np.array([[0, 1], [0, 1]])
    assert compare_arrays(known_pts, network._pts)
    known_edges = np.array([[0], [1]])
    assert compare_arrays(known_edges, network._edges)

    # Then load no data
    network = fracture_importer._network_2d_from_csv(file_name, max_num_fracs=0)
    assert network._pts.shape == (2, 0)
    assert network._edges.shape == (2, 0)
    assert network.domain is None
    assert network.num_frac() == 0


def test_domain_assignment(file_name):
    p = np.array([0, 0, 1, 1])
    f = np.hstack((0, p))
    np.savetxt(file_name, f, delimiter=",")
    domain = pp.Domain({"xmin": -1, "xmax": 0, "ymin": -2, "ymax": 2})

    network = fracture_importer._network_2d_from_csv(file_name, domain=domain)

    assert network.domain.bounding_box["xmin"] == -1
    assert network.domain.bounding_box["ymin"] == -2
    assert network.domain.bounding_box["xmax"] == 0
    assert network.domain.bounding_box["ymax"] == 2


def test_polyline_single_branch(file_name):
    p = np.array([[0, 0], [1, 1]])
    frac_id = 0
    f = np.hstack((frac_id * np.ones(2).reshape((-1, 1)), p))
    np.savetxt(file_name, f, delimiter=",")

    network, fid = fracture_importer._network_2d_from_csv(
        file_name, polyline=True, return_frac_id=True
    )
    known_pts = np.array([[0, 1], [0, 1]])
    assert compare_arrays(known_pts, network._pts)
    known_edges = np.array([[0], [1]])
    assert compare_arrays(known_edges, network._edges)
    assert network.domain.bounding_box["xmin"] == 0
    assert network.domain.bounding_box["ymin"] == 0
    assert network.domain.bounding_box["xmax"] == 1
    assert network.domain.bounding_box["ymax"] == 1

    assert fid.size == 1
    assert fid[0] == frac_id


def test_polyline_two_branches(file_name):
    p = np.array([[0, 0], [1, 1], [2, 2]])
    frac_id = 1
    f = np.hstack((frac_id * np.ones(3).reshape((-1, 1)), p))
    np.savetxt(file_name, f, delimiter=",")

    network, fid = fracture_importer._network_2d_from_csv(
        file_name, polyline=True, return_frac_id=True
    )
    known_pts = np.array([[0, 1, 2], [0, 1, 2]])
    assert compare_arrays(known_pts, network._pts)
    known_edges = np.array([[0, 1], [1, 2]])
    assert compare_arrays(known_edges, network._edges)

    assert fid.size == 2
    assert np.all(fid == frac_id)


def test_polyline_two_fractures(file_name):
    p = np.array([[0, 0], [1, 1], [2, 2], [4, 4], [5, 5]])
    frac_id_1 = 1
    frac_id_2 = 2
    f = np.hstack(
        (
            np.hstack((frac_id_1 * np.ones(3), frac_id_2 * np.ones(2))).reshape(
                (-1, 1)
            ),
            p,
        )
    )
    np.savetxt(file_name, f, delimiter=",")

    network, fid = fracture_importer._network_2d_from_csv(
        file_name, polyline=True, return_frac_id=True
    )
    known_pts = np.array([[0, 1, 2, 4, 5], [0, 1, 2, 4, 5]])
    assert compare_arrays(known_pts, network._pts)
    known_edges = np.array([[0, 1, 3], [1, 2, 4]])
    assert compare_arrays(known_edges, network._edges)

    assert fid.size == 3
    assert np.all(fid[:2] == frac_id_1)
    assert np.all(fid[2:] == frac_id_2)


# ---------- Testing network_3d_from_csv and elliptic_network_3d_from_csv ----------


@pytest.fixture(
    params=[
        fracture_importer._network_3d_from_csv,
        fracture_importer.elliptic_network_3d_from_csv,
    ]
)
def make_network_3d_from_csv(request) -> Callable[[Path], FractureNetwork3d]:
    return request.param


def test_domain_only(file_name, make_network_3d_from_csv):
    domain = np.atleast_2d(np.array([0, 1, 2, 3, 4, 5]))
    np.savetxt(file_name, domain, delimiter=",")

    network = make_network_3d_from_csv(file_name)
    assert len(network.fractures) == 0
    assert network.domain.bounding_box["xmin"] == 0
    assert network.domain.bounding_box["ymin"] == 1
    assert network.domain.bounding_box["zmin"] == 2
    assert network.domain.bounding_box["xmax"] == 3
    assert network.domain.bounding_box["ymax"] == 4
    assert network.domain.bounding_box["zmax"] == 5


def test_single_fracture(file_name):
    p = np.atleast_2d(np.array([0, 0, 0, 1, 1, 1, 1, 0, 1]))
    np.savetxt(file_name, p, delimiter=",")

    network = fracture_importer._network_3d_from_csv(file_name, has_domain=False)
    known_p = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 1]])
    assert len(network.fractures) == 1
    assert compare_arrays(known_p, network.fractures[0].pts)


def test_two_fractures(file_name):
    # Two fractures, identical coordinates - this will not matter
    p = np.atleast_2d(
        np.array([[0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 1]])
    )
    np.savetxt(file_name, p, delimiter=",")

    network = fracture_importer._network_3d_from_csv(file_name, has_domain=False)
    known_p = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 1]])
    assert len(network.fractures) == 2
    assert compare_arrays(known_p, network.fractures[0].pts)
    assert compare_arrays(known_p, network.fractures[1].pts)


def test_create_fracture_elliptic(file_name):
    p = np.atleast_2d([0, 0, 0, 2, 1, 0, 0, 0, 16])
    np.savetxt(file_name, p, delimiter=",")

    network = pp.fracture_importer.elliptic_network_3d_from_csv(
        file_name, has_domain=False
    )

    assert len(network.fractures) == 1
    f = network.fractures[0]
    assert compare_arrays(f.center, np.zeros((3, 1)))
    assert f.pts.shape[1] == 16
    assert f.pts[0].max() == 2
    assert f.pts[1].max() == 1
    assert f.pts[2].max() == 0
    assert f.pts[0].min() == -2
    assert f.pts[1].min() == -1
    assert f.pts[2].min() == 0


# ---------- Testing network_2d_from_csv with DFN model ----------


def test_one_fracture_dfn(file_name):
    p = np.array([0, 0, 1, 1])
    f = np.hstack((0, p))
    np.savetxt(file_name, f, delimiter=",")

    network = fracture_importer._network_2d_from_csv(file_name)

    mesh_args = {"mesh_size_frac": 0.3, "mesh_size_bound": 0.3}
    mdg = network.mesh(mesh_args, dfn=True)

    bmin, bmax = pp.domain.mdg_minmax_coordinates(mdg)
    assert np.allclose(bmin, [0, 0, 0])
    assert np.allclose(bmax, [1, 1, 0])

    assert mdg.dim_max() == 1
    assert mdg.dim_min() == 1
    assert mdg.num_subdomains() == 1
    assert mdg.num_interfaces() == 0


def test_two_fractures_dfn(file_name):
    p = np.array([[0, 0, 1, 0.45], [0, 1, 1, 1]])
    f = np.hstack(([[0], [1]], p))
    np.savetxt(file_name, f, delimiter=",")

    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1.1})
    network = fracture_importer._network_2d_from_csv(file_name, domain=domain)
    mesh_args = {"mesh_size_frac": 0.2, "mesh_size_bound": 0.2}
    mdg = network.mesh(mesh_args, dfn=True)

    bmin, bmax = pp.domain.mdg_minmax_coordinates(mdg)
    assert np.allclose(bmin, [0, 0, 0])
    assert np.allclose(bmax, [1, 1, 0])

    assert mdg.dim_max() == 1
    assert mdg.dim_min() == 1
    assert mdg.num_subdomains() == 2
    assert mdg.num_interfaces() == 0

    for sd in mdg.subdomains():
        _, bmax = pp.domain.grid_minmax_coordinates(sd)
        assert np.allclose(bmax, [1, 0.45, 0]) ^ np.allclose(bmax, [1, 1, 0])


def test_two_intersecting_fractures_dfn():
    p = np.array([[0, 0, 1, 0.5], [0, 1, 1, 0]])
    f = np.hstack(([[0], [1]], p))
    file_name = Path("frac.csv")
    np.savetxt(file_name, f, delimiter=",")

    network = fracture_importer._network_2d_from_csv(file_name)
    mesh_args = {"mesh_size_frac": 0.2, "mesh_size_bound": 0.2}
    mdg = network.mesh(mesh_args, dfn=True)

    assert mdg.dim_max() == 1
    assert mdg.dim_min() == 0
    assert mdg.num_subdomains() == 3
    assert mdg.num_interfaces() == 2

    for sd in mdg.subdomains():
        _, bmax = pp.domain.grid_minmax_coordinates(sd)
        if sd.dim == 1:
            assert np.allclose(bmax, [1, 0.5, 0]) ^ np.allclose(bmax, [1, 1, 0])
        elif sd.dim == 0:
            assert np.allclose(bmax, [0.66666667, 0.33333333, 0])
        else:
            assert False
