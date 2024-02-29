"""Testing functionality related to the fracture importer. These functions are covered:
- network_2d_from_csv
- network_3d_from_csv
- elliptic_network_3d_from_csv

Created on Wed Dec 12 09:05:31 2018

@author: eke001
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs import fracture_importer
from porepy.fracs.fracture_network_3d import FractureNetwork3d

# ---------- Testing network_2d_from_csv ----------


@pytest.fixture
def file_name() -> str:
    yield (file_name := "frac.csv")

    # Tear down
    Path(file_name).unlink(missing_ok=True)


def test_single_fracture_2d(file_name):
    p = np.array([0, 0, 1, 1])
    f = np.hstack((0, p))
    np.savetxt(file_name, f, delimiter=",")

    network = fracture_importer.network_2d_from_csv(file_name, skip_header=0)
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

    network, fid = fracture_importer.network_2d_from_csv(
        file_name, skip_header=0, return_frac_id=True
    )

    assert fid.size == 1
    assert fid[0] == frac_id


def test_no_data(file_name):
    np.savetxt(file_name, [], delimiter=",")
    network = fracture_importer.network_2d_from_csv(file_name, skip_header=0)
    assert network._pts.shape == (2, 0)
    assert network._edges.shape == (2, 0)
    assert network.domain is None
    assert network.num_frac() == 0


def test_max_num_fracs_keyword(file_name):
    p = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    f = np.hstack((np.arange(2).reshape((-1, 1)), p))
    np.savetxt(file_name, f, delimiter=",")

    # First load one fracture only
    network = fracture_importer.network_2d_from_csv(
        file_name, skip_header=0, max_num_fracs=1
    )
    known_pts = np.array([[0, 1], [0, 1]])
    assert compare_arrays(known_pts, network._pts)
    known_edges = np.array([[0], [1]])
    assert compare_arrays(known_edges, network._edges)

    # Then load no data
    network = fracture_importer.network_2d_from_csv(
        file_name, skip_header=0, max_num_fracs=0
    )
    assert network._pts.shape == (2, 0)
    assert network._edges.shape == (2, 0)
    assert network.domain is None
    assert network.num_frac() == 0


def test_domain_assignment(file_name):
    p = np.array([0, 0, 1, 1])
    f = np.hstack((0, p))
    np.savetxt(file_name, f, delimiter=",")
    domain = pp.Domain({"xmin": -1, "xmax": 0, "ymin": -2, "ymax": 2})

    network = fracture_importer.network_2d_from_csv(
        file_name, skip_header=0, domain=domain
    )

    assert network.domain.bounding_box["xmin"] == -1
    assert network.domain.bounding_box["ymin"] == -2
    assert network.domain.bounding_box["xmax"] == 0
    assert network.domain.bounding_box["ymax"] == 2


def test_polyline_single_branch(file_name):
    p = np.array([[0, 0], [1, 1]])
    frac_id = 0
    f = np.hstack((frac_id * np.ones(2).reshape((-1, 1)), p))
    np.savetxt(file_name, f, delimiter=",")

    network, fid = fracture_importer.network_2d_from_csv(
        file_name, skip_header=0, polyline=True, return_frac_id=True
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

    network, fid = fracture_importer.network_2d_from_csv(
        file_name, skip_header=0, polyline=True, return_frac_id=True
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

    network, fid = fracture_importer.network_2d_from_csv(
        file_name, skip_header=0, polyline=True, return_frac_id=True
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
        fracture_importer.network_3d_from_csv,
        fracture_importer.elliptic_network_3d_from_csv,
    ]
)
def make_network_3d_from_csv(request) -> Callable[[str], FractureNetwork3d]:
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

    network = fracture_importer.network_3d_from_csv(file_name, has_domain=False)
    known_p = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 1]])
    assert len(network.fractures) == 1
    assert compare_arrays(known_p, network.fractures[0].pts)


def test_two_fractures(file_name):
    # Two fractures, identical coordinates - this will not matter
    p = np.atleast_2d(
        np.array([[0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 1]])
    )
    np.savetxt(file_name, p, delimiter=",")

    network = fracture_importer.network_3d_from_csv(file_name, has_domain=False)
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

    network = fracture_importer.network_2d_from_csv(file_name, skip_header=0)

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
    network = fracture_importer.network_2d_from_csv(
        file_name, domain=domain, skip_header=0
    )
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
    file_name = "frac.csv"
    np.savetxt(file_name, f, delimiter=",")

    network = fracture_importer.network_2d_from_csv(file_name, skip_header=0)
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
