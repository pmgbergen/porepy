#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module contains completed setups for simple contact mechanics problems,
with and without poroelastic effects in the deformation of the Nd domain.
"""
import logging

import numpy as np
from scipy.spatial.distance import cdist

import porepy as pp
from porepy.models import contact_mechanics_model

# Module-wide logger
logger = logging.getLogger(__name__)


class ContactMechanicsExample(contact_mechanics_model.ContactMechanics):
    def __init__(self, mesh_args, folder_name, params=None):
        self.mesh_args = mesh_args
        self.folder_name = folder_name

        super().__init__(params)

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            frac_pts (np.array): Nd x (number of fracture points), the coordinates of
                the fracture endpoints.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
            gb (pp.GridBucket): The produced grid bucket.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.

        """
        x_endpoints = np.array([0.2, 0.8])
        gb, self.box = gb, self.box = pp.grid_buckets_2d.single_horizontal(
            self.mesh_args,
            x_endpoints,
        )

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self._Nd = self.gb.dim_max()

    def _domain_boundary_sides(self, g):
        """
        Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries.
        """
        tol = 1e-10
        box = self.box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        north = g.face_centers[1] > box["ymax"] - tol
        south = g.face_centers[1] < box["ymin"] + tol
        if self._Nd == 2:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom

    def _set_friction_coefficient(self, g):

        nodes = g.nodes

        tips = nodes[:, [0, -1]]

        fc = g.cell_centers
        D = cdist(fc.T, tips.T)
        D = np.min(D, axis=1)
        R = 200
        beta = 10
        friction_coefficient = 0.5 * (1 + beta * np.exp(-R * D ** 2))
        return friction_coefficient


class ProblemDataTime:
    """
    This class contains the problem specific methods for the Biot and THM
    integration tests. Model specific methods are inherited from the respective
    model classes.
    """

    def create_grid(self):
        """
        Method that creates and returns the GridBucket of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        with_fracture = getattr(self, "with_fracture", True)
        simplex = getattr(self, "simplex", True)
        if with_fracture:
            gb, self.box = pp.grid_buckets_2d.single_horizontal(
                self.mesh_args, simplex=simplex
            )
            pp.contact_conditions.set_projections(gb)
        else:
            nx = getattr(self, "nx", [3, 3])
            gb = pp.meshing.cart_grid([], nx, physdims=[1, 1])
            self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        self.gb = gb
        self._Nd = gb.dim_max()

    def _source_scalar(self, g):
        if g.dim == self._Nd:
            values = np.zeros(g.num_cells)
        else:
            values = self.scalar_source_value * np.ones(g.num_cells)
        return values

    def _bc_type_mechanics(self, g):
        _, _, _, north, south, _, _ = self._domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, north + south, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def _bc_type_scalar(self, g):
        _, _, _, north, south, _, _ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, north + south, "dir")

    def _bc_type_temperature(self, g):
        _, _, _, north, south, _, _ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, north + south, "dir")

    def _bc_values_mechanics(self, g):
        # Set the boundary values
        _, _, _, north, south, _, _ = self._domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))

        values[0, south] = self.ux_south * (self.time > 0.1)
        values[1, south] = self.uy_south * (self.time > 0.1)
        values[0, north] = self.ux_north * (self.time > 0.1)
        values[1, north] = self.uy_north * (self.time > 0.1)
        return values.ravel("F")

    def _compute_aperture(self, g, from_iterate=True):
        self.initial_aperture = 1e-4
        apertures = np.ones(g.num_cells)
        gb = self.gb
        if g.dim == (self._Nd - 1):
            # Initial aperture
            apertures *= self.initial_aperture

            data = gb.node_props(g)
            proj = data["tangential_normal_projection"]

            # Reconstruct the displacement solution on the fracture
            g_h = gb.node_neighbors(g)[0]
            assert g_h.dim == self._Nd
            data_edge = gb.edge_props((g, g_h))
            if pp.STATE in data_edge:
                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, projection=proj, from_iterate=from_iterate
                )
                # Magnitudes of normal and tangential components
                norm_u_n = np.absolute(u_mortar_local[-1])
                # Add contributions
                apertures += norm_u_n

        return apertures
