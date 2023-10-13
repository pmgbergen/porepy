import unittest
import pytest

import numpy as np

import porepy as pp
from porepy.fracs.utils import pts_edges_to_linefractures
from porepy.grids.standard_grids.utils import unit_domain

"""
In this test we validate the propagation of physical tags from gmsh to porepy. We 
consider the case with only boundary, fractures, auxiliary segments, and a mixed of
them.
"""


def check_boundary_faces(g: pp.Grid, num_auxiliary: int):
    tag = np.where(g.tags[f"domain_boundary_line_{num_auxiliary}_faces"])[0]
    assert np.allclose(g.face_centers[1, tag], 0)

    tag = np.where(g.tags[f"domain_boundary_line_{num_auxiliary + 1}_faces"])[0]
    assert np.allclose(g.face_centers[0, tag], 1)

    tag = np.where(g.tags[f"domain_boundary_line_{num_auxiliary + 2}_faces"])[0]
    assert np.allclose(g.face_centers[1, tag], 1)

    tag = np.where(g.tags[f"domain_boundary_line_{num_auxiliary + 3}_faces"])[0]
    assert np.allclose(g.face_centers[0, tag], 0)


def check_auxiliary_fracture_faces(
    g: pp.Grid,
    key: str,
    start: list[np.ndarray],
    end: list[np.ndarray],
    offset: int = 0,
):
    num_lines = len(start)
    for i in range(num_lines):
        tag = g.tags[f"{key}_{i + offset}_faces"]
        dist, _ = pp.distances.points_segments(g.face_centers[:2], start[i], end[i])

        assert np.allclose(dist[tag, 0], 0)
        assert np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))


class TestGmshTags:
    def test_face_tags_from_gmsh_before_grid_split(self):
        """Check that the faces of the grids returned from gmsh has correct tags
        before the grid is split. The domain is the unit square with two fractures
        that intersect.
        """
        p = np.array([[0, 1, 0.5, 0.5], [0.5, 0.5, 0, 1]])
        e = np.array([[0, 2], [1, 3]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))
        mesh_args = {
            "mesh_size_frac": 0.1,
            "mesh_size_bound": 0.1,
        }

        file_name = "mesh_simplex.msh"
        gmsh_data = network.prepare_for_gmsh(mesh_args, None, True, None, False)

        # Consider the dimension of the problem
        ndim = 2
        gmsh_writer = pp.fracs.gmsh_interface.GmshWriter(gmsh_data)
        gmsh_writer.generate(file_name, ndim)

        grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(
            file_name,
        )
        pp.meshing._tag_faces(grid_list, False)
        for grid_of_dim in grid_list:
            for g in grid_of_dim:
                g.compute_geometry()
                if g.dim == 2:
                    f_cc0 = g.face_centers[:, g.tags["fracture_0_faces"]]
                    f_cc1 = g.face_centers[:, g.tags["fracture_1_faces"]]

                    # Test position of the fracture faces. Fracture 1 is aligned with
                    # the y-axis, fracture 2 is aligned with the x-axis
                    assert np.allclose(f_cc0[1], 0.5)
                    assert np.allclose(f_cc1[0], 0.5)

                if g.dim > 0:
                    db_cc = g.face_centers[:, g.tags["domain_boundary_faces"]]
                    # Test that domain boundary faces are on the boundary
                    assert np.all(
                        (np.abs(db_cc[0]) < 1e-10)
                        | (np.abs(db_cc[1]) < 1e-10)
                        | (np.abs(db_cc[0] - 1) < 1e-10)
                        | (np.abs(db_cc[1] - 1) < 1e-10)
                    )
                # No tip faces in this test case
                assert np.all(g.tags["tip_faces"] == 0)

    def test_boundary(self):
        """No fractures, test that the boundary faces are correct."""
        network = pp.create_fracture_network(None, unit_domain(2))
        mesh_args = {"mesh_size_frac": 1}
        mdg = network.mesh(mesh_args)
        g = mdg.subdomains(dim=2)[0]
        check_boundary_faces(g, 0)

    @pytest.mark.parametrize(
        "mesh_args", [{"mesh_size_frac": 1}, {"mesh_size_frac": 0.33}]
    )
    def test_single_constraint(self, mesh_args):
        """A single auxiliary line (constraints)"""
        subdomain_start = np.array([[0.25], [0.5]])
        subdomain_end = np.array([[0.5], [0.5]])
        p = np.hstack((subdomain_start, subdomain_end))
        e = np.array([[0], [1]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))

        mesh_args = {"mesh_size_frac": 1}
        mdg = network.mesh(mesh_args, constraints=np.array([0]))
        g = mdg.subdomains(dim=2)[0]

        check_boundary_faces(g, 1)
        check_auxiliary_fracture_faces(
            g, "auxiliary_line", [subdomain_start], [subdomain_end]
        )

    @pytest.mark.parametrize(
        "mesh_args", [{"mesh_size_frac": 1}, {"mesh_size_frac": 0.25}]
    )
    def test_two_constraints(self, mesh_args):
        constraint_start_0 = np.array([[0.0], [0.5]])
        constraint_end_0 = np.array([[0.75], [0.5]])
        constraint_start_1 = np.array([[0.5], [0.25]])
        constraint_end_1 = np.array([[0.5], [0.75]])
        p = np.hstack(
            (constraint_start_0, constraint_end_0, constraint_start_1, constraint_end_1)
        )
        e = np.array([[0, 2], [1, 3]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))

        mdg = network.mesh(mesh_args, constraints=np.arange(2))
        g = mdg.subdomains(dim=2)[0]

        check_boundary_faces(g, 2)

        check_auxiliary_fracture_faces(
            g,
            "auxiliary_line",
            [constraint_start_0, constraint_start_1],
            [constraint_end_0, constraint_end_1],
        )

    @pytest.mark.parametrize(
        "mesh_args", [{"mesh_size_frac": 1}, {"mesh_size_frac": 0.25}]
    )
    def test_single_fracture(self, mesh_args):
        frac_start = np.array([[0.25], [0.5]])
        frac_end = np.array([[0.75], [0.5]])
        mdg, _ = pp.md_grids_2d.single_horizontal(
            mesh_args, x_endpoints=np.array([0.25, 0.75])
        )
        g = mdg.subdomains(dim=2)[0]

        check_boundary_faces(g, 1)

        check_auxiliary_fracture_faces(g, "fracture", [frac_start], [frac_end])

    def test_fracture_and_constraints(self):
        frac_start = np.array([[0.5], [0.25]])
        frac_end = np.array([[0.5], [0.75]])
        constraint_start = np.array([[0.25], [0.5]])
        constraint_end = np.array([[0.5], [0.5]])
        p = np.hstack((frac_start, frac_end, constraint_start, constraint_end))
        e = np.array([[0, 2], [1, 3]])
        fractures = pts_edges_to_linefractures(p, e)
        network = pp.create_fracture_network(fractures, unit_domain(2))

        constraints = np.array([1])
        mesh_args = {"mesh_size_frac": 1}
        mdg = network.mesh(mesh_args, constraints=constraints)
        g = mdg.subdomains(dim=2)[0]

        check_boundary_faces(g, 2)

        check_auxiliary_fracture_faces(g, "fracture", [frac_start], [frac_end])
        check_auxiliary_fracture_faces(
            g, "auxiliary_line", [constraint_start], [constraint_end], 1
        )
