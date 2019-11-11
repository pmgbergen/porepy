#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module contains completed setups for simple contact mechanics problems,
with and without poroelastic effects in the deformation of the Nd domain.
"""
import numpy as np
from scipy.spatial.distance import cdist
import logging

import porepy as pp
import porepy.models.contact_mechanics_model
import porepy.models.contact_mechanics_biot_model
import porepy.utils.derived_discretizations.implicit_euler as IE_discretizations

# Module-wide logger
logger = logging.getLogger(__name__)


class ContactMechanicsExample(pp.models.contact_mechanics_model.ContactMechanics):
    def __init__(self, mesh_args, folder_name, params=None):
        self.mesh_args = mesh_args
        self.folder_name = folder_name

        super(ContactMechanicsExample, self).__init__(params)

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
        # List the fracture points
        self.frac_pts = np.array([[0.2, 0.8], [0.5, 0.5]])
        # Each column defines one fracture
        frac_edges = np.array([[0], [1]])
        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()

    def domain_boundary_sides(self, g):
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
        if self.Nd == 2:
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


class ContactMechanicsBiotExample(
    porepy.models.contact_mechanics_biot_model.ContactMechanicsBiot
):
    def __init__(self, mesh_args, folder_name, params=None):
        super().__init__(params)
        self.mesh_args = mesh_args
        self.folder_name = folder_name

        # Time
        # The time attribute may be used e.g. to update BCs.
        self.time = 0
        self.time_step = 1 * self.length_scale ** 2
        self.end_time = self.time_step * 1

        # Whether or not to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure = True

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
        # List the fracture points
        self.frac_pts = np.array([[0.2, 0.8], [0.5, 0.5]])
        # Each column defines one fracture
        frac_edges = np.array([[0], [1]])
        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()

    def domain_boundary_sides(self, g):
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
        if self.Nd == 2:
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
