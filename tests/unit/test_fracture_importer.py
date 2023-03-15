#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:05:31 2018

@author: eke001
"""

import unittest

import numpy as np

import porepy as pp
from tests import test_utils


class TestImport2dCsv(unittest.TestCase):
    def test_single_fracture(self):
        p = np.array([0, 0, 1, 1])
        f = np.hstack((0, p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        network = pp.fracture_importer.network_2d_from_csv(file_name, skip_header=0)
        known_pts = np.array([[0, 1], [0, 1]])
        self.assertTrue(test_utils.compare_arrays(known_pts, network._pts))
        known_edges = np.array([[0], [1]])
        self.assertTrue(test_utils.compare_arrays(known_edges, network._edges))
        self.assertTrue(network.domain.bounding_box["xmin"] == 0)
        self.assertTrue(network.domain.bounding_box["ymin"] == 0)
        self.assertTrue(network.domain.bounding_box["xmax"] == 1)
        self.assertTrue(network.domain.bounding_box["ymax"] == 1)

        test_utils.delete_file(file_name)

    def test_return_frac_id(self):
        p = np.array([0, 0, 1, 1])
        frac_id = np.random.randint(0, 10)
        f = np.hstack((frac_id, p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        network, fid = pp.fracture_importer.network_2d_from_csv(
            file_name, skip_header=0, return_frac_id=True
        )

        self.assertTrue(fid.size == 1)
        self.assertTrue(fid[0] == frac_id)
        test_utils.delete_file(file_name)

    def test_no_data(self):
        file_name = "frac.csv"
        np.savetxt(file_name, [], delimiter=",")
        network = pp.fracture_importer.network_2d_from_csv(file_name, skip_header=0)
        self.assertTrue(network._pts.shape == (2, 0))
        self.assertTrue(network._edges.shape == (2, 0))
        self.assertTrue(network.domain is None)
        self.assertTrue(network.num_frac() == 0)
        test_utils.delete_file(file_name)

    def test_max_num_fracs_keyword(self):
        p = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
        f = np.hstack((np.arange(2).reshape((-1, 1)), p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        # First load one fracture only
        network = pp.fracture_importer.network_2d_from_csv(
            file_name, skip_header=0, max_num_fracs=1
        )
        known_pts = np.array([[0, 1], [0, 1]])
        self.assertTrue(test_utils.compare_arrays(known_pts, network._pts))
        known_edges = np.array([[0], [1]])
        self.assertTrue(test_utils.compare_arrays(known_edges, network._edges))

        # Then load no data
        network = pp.fracture_importer.network_2d_from_csv(
            file_name, skip_header=0, max_num_fracs=0
        )
        self.assertTrue(network._pts.shape == (2, 0))
        self.assertTrue(network._edges.shape == (2, 0))
        self.assertTrue(network.domain is None)
        self.assertTrue(network.num_frac() == 0)

        test_utils.delete_file(file_name)

    def test_domain_assignment(self):
        p = np.array([0, 0, 1, 1])
        f = np.hstack((0, p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")
        domain = {"xmin": -1, "xmax": 0, "ymin": -2, "ymax": 2}

        network = pp.fracture_importer.network_2d_from_csv(
            file_name, skip_header=0, domain=domain
        )

        self.assertTrue(network.domain["xmin"] == -1)
        self.assertTrue(network.domain["ymin"] == -2)
        self.assertTrue(network.domain["xmax"] == 0)
        self.assertTrue(network.domain["ymax"] == 2)

        test_utils.delete_file(file_name)

    def test_polyline_single_branch(self):
        p = np.array([[0, 0], [1, 1]])
        frac_id = 0
        f = np.hstack((frac_id * np.ones(2).reshape((-1, 1)), p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        network, fid = pp.fracture_importer.network_2d_from_csv(
            file_name, skip_header=0, polyline=True, return_frac_id=True
        )
        known_pts = np.array([[0, 1], [0, 1]])
        self.assertTrue(test_utils.compare_arrays(known_pts, network._pts))
        known_edges = np.array([[0], [1]])
        self.assertTrue(test_utils.compare_arrays(known_edges, network._edges))
        self.assertTrue(network.domain.bounding_box["xmin"] == 0)
        self.assertTrue(network.domain.bounding_box["ymin"] == 0)
        self.assertTrue(network.domain.bounding_box["xmax"] == 1)
        self.assertTrue(network.domain.bounding_box["ymax"] == 1)

        self.assertTrue(fid.size == 1)
        self.assertTrue(fid[0] == frac_id)

        test_utils.delete_file(file_name)

    def test_polyline_two_branches(self):
        p = np.array([[0, 0], [1, 1], [2, 2]])
        frac_id = 1
        f = np.hstack((frac_id * np.ones(3).reshape((-1, 1)), p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        network, fid = pp.fracture_importer.network_2d_from_csv(
            file_name, skip_header=0, polyline=True, return_frac_id=True
        )
        known_pts = np.array([[0, 1, 2], [0, 1, 2]])
        self.assertTrue(test_utils.compare_arrays(known_pts, network._pts))
        known_edges = np.array([[0, 1], [1, 2]])
        self.assertTrue(test_utils.compare_arrays(known_edges, network._edges))

        self.assertTrue(fid.size == 2)
        self.assertTrue(np.all(fid == frac_id))

        test_utils.delete_file(file_name)

    def test_polyline_two_fractures(self):
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
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        network, fid = pp.fracture_importer.network_2d_from_csv(
            file_name, skip_header=0, polyline=True, return_frac_id=True
        )
        known_pts = np.array([[0, 1, 2, 4, 5], [0, 1, 2, 4, 5]])
        self.assertTrue(test_utils.compare_arrays(known_pts, network._pts))
        known_edges = np.array([[0, 1, 3], [1, 2, 4]])
        self.assertTrue(test_utils.compare_arrays(known_edges, network._edges))

        self.assertTrue(fid.size == 3)
        self.assertTrue(np.all(fid[:2] == frac_id_1))
        self.assertTrue(np.all(fid[2:] == frac_id_2))

        test_utils.delete_file(file_name)


class TestImport3dCsv(unittest.TestCase):
    def test_domain_only(self):
        domain = np.atleast_2d(np.array([0, 1, 2, 3, 4, 5]))
        file_name = "frac.csv"
        np.savetxt(file_name, domain, delimiter=",")

        network = pp.fracture_importer.network_3d_from_csv(file_name)
        self.assertTrue(len(network.fractures) == 0)
        self.assertTrue(network.domain.bounding_box["xmin"] == 0)
        self.assertTrue(network.domain.bounding_box["ymin"] == 1)
        self.assertTrue(network.domain.bounding_box["zmin"] == 2)
        self.assertTrue(network.domain.bounding_box["xmax"] == 3)
        self.assertTrue(network.domain.bounding_box["ymax"] == 4)
        self.assertTrue(network.domain.bounding_box["zmax"] == 5)

    def test_single_fracture(self):
        p = np.atleast_2d(np.array([0, 0, 0, 1, 1, 1, 1, 0, 1]))
        file_name = "frac.csv"
        np.savetxt(file_name, p, delimiter=",")

        network = pp.fracture_importer.network_3d_from_csv(file_name, has_domain=False)
        known_p = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 1]])
        self.assertTrue(len(network.fractures) == 1)
        self.assertTrue(test_utils.compare_arrays(known_p, network.fractures[0].pts))

    def test_two_fractures(self):
        # Two fractures, identical coordinates - this will not matter
        p = np.atleast_2d(
            np.array([[0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 1]])
        )
        file_name = "frac.csv"
        np.savetxt(file_name, p, delimiter=",")

        network = pp.fracture_importer.network_3d_from_csv(file_name, has_domain=False)
        known_p = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 1]])
        self.assertTrue(len(network.fractures) == 2)
        self.assertTrue(test_utils.compare_arrays(known_p, network.fractures[0].pts))
        self.assertTrue(test_utils.compare_arrays(known_p, network.fractures[1].pts))


class TestImport3dElliptic(unittest.TestCase):
    def test_domain_only(self):
        domain = np.atleast_2d(np.array([0, 1, 2, 3, 4, 5]))
        file_name = "frac.csv"
        np.savetxt(file_name, domain, delimiter=",")

        network = pp.fracture_importer.network_3d_from_csv(file_name)
        self.assertTrue(len(network.fractures) == 0)
        self.assertTrue(network.domain.bounding_box["xmin"] == 0)
        self.assertTrue(network.domain.bounding_box["ymin"] == 1)
        self.assertTrue(network.domain.bounding_box["zmin"] == 2)
        self.assertTrue(network.domain.bounding_box["xmax"] == 3)
        self.assertTrue(network.domain.bounding_box["ymax"] == 4)
        self.assertTrue(network.domain.bounding_box["zmax"] == 5)

    def test_create_fracture(self):
        p = np.atleast_2d([0, 0, 0, 2, 1, 0, 0, 0, 16])
        file_name = "frac.csv"
        np.savetxt(file_name, p, delimiter=",")

        network = pp.fracture_importer.elliptic_network_3d_from_csv(
            file_name, has_domain=False
        )

        self.assertTrue(len(network.fractures) == 1)
        f = network.fractures[0]
        self.assertTrue(test_utils.compare_arrays(f.center, np.zeros((3, 1))))
        self.assertTrue(f.pts.shape[1] == 16)
        self.assertTrue(f.pts[0].max() == 2)
        self.assertTrue(f.pts[1].max() == 1)
        self.assertTrue(f.pts[2].max() == 0)
        self.assertTrue(f.pts[0].min() == -2)
        self.assertTrue(f.pts[1].min() == -1)
        self.assertTrue(f.pts[2].min() == 0)


class TestImportDFN1d(unittest.TestCase):
    def test_one_fracture(self):
        p = np.array([0, 0, 1, 1])
        f = np.hstack((0, p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        network = pp.fracture_importer.network_2d_from_csv(file_name, skip_header=0)

        mesh_args = {"mesh_size_frac": 0.3, "mesh_size_bound": 0.3}
        mdg = network.mesh(mesh_args, dfn=True)

        bmin, bmax = pp.domain.mdg_minmax_coordinates(mdg)
        self.assertTrue(np.allclose(bmin, [0, 0, 0]))
        self.assertTrue(np.allclose(bmax, [1, 1, 0]))

        self.assertTrue(mdg.dim_max() == 1)
        self.assertTrue(mdg.dim_min() == 1)
        self.assertTrue(mdg.num_subdomains() == 1)
        self.assertTrue(mdg.num_interfaces() == 0)

    def test_two_fractures(self):
        p = np.array([[0, 0, 1, 0.45], [0, 1, 1, 1]])
        f = np.hstack(([[0], [1]], p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1.1})
        network = pp.fracture_importer.network_2d_from_csv(
            file_name, domain=domain, skip_header=0
        )
        mesh_args = {"mesh_size_frac": 0.2, "mesh_size_bound": 0.2}
        mdg = network.mesh(mesh_args, dfn=True)

        bmin, bmax = pp.domain.mdg_minmax_coordinates(mdg)
        self.assertTrue(np.allclose(bmin, [0, 0, 0]))
        self.assertTrue(np.allclose(bmax, [1, 1, 0]))

        self.assertTrue(mdg.dim_max() == 1)
        self.assertTrue(mdg.dim_min() == 1)
        self.assertTrue(mdg.num_subdomains() == 2)
        self.assertTrue(mdg.num_interfaces() == 0)

        for sd in mdg.subdomains():
            _, bmax = pp.domain.grid_minmax_coordinates(sd)
            self.assertTrue(
                np.allclose(bmax, [1, 0.45, 0]) ^ np.allclose(bmax, [1, 1, 0])
            )

    def test_two_intersecting_fractures(self):
        p = np.array([[0, 0, 1, 0.5], [0, 1, 1, 0]])
        f = np.hstack(([[0], [1]], p))
        file_name = "frac.csv"
        np.savetxt(file_name, f, delimiter=",")

        network = pp.fracture_importer.network_2d_from_csv(file_name, skip_header=0)
        mesh_args = {"mesh_size_frac": 0.2, "mesh_size_bound": 0.2}
        mdg = network.mesh(mesh_args, dfn=True)

        self.assertTrue(mdg.dim_max() == 1)
        self.assertTrue(mdg.dim_min() == 0)
        self.assertTrue(mdg.num_subdomains() == 3)
        self.assertTrue(mdg.num_interfaces() == 2)

        for sd in mdg.subdomains():
            _, bmax = pp.domain.grid_minmax_coordinates(sd)
            if sd.dim == 1:
                self.assertTrue(
                    np.allclose(bmax, [1, 0.5, 0]) ^ np.allclose(bmax, [1, 1, 0])
                )
            elif sd.dim == 0:
                self.assertTrue(np.allclose(bmax, [0.66666667, 0.33333333, 0]))
            else:
                self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
