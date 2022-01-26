#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains utility functions for setting up the problems related to a 2d benhcmark
as described by Flemisch et al (2018).
"""

import numpy as np

import porepy as pp


def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param", "is_tangential"])
    tol = 1e-5
    a = 1e-4

    for g, d in gb:
        # Assign aperture
        a_dim = np.power(a, gb.dim_max() - g.dim)
        aperture = np.ones(g.num_cells) * a_dim

        # Effective permeability, scaled with aperture.
        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max()) * aperture
        if g.dim == 2:
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=1, kzz=1)

        specified_parameters = {"second_order_tensor": perm}
        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[right] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = -a_dim * g.face_areas[bound_faces[left]]
            bc_val[bound_faces[right]] = 1

            bound = pp.BoundaryCondition(g, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
        else:
            bound = pp.BoundaryCondition(g, np.empty(0), np.empty(0))
            specified_parameters.update({"bc": bound})

        d["is_tangential"] = True
        pp.initialize_default_data(g, d, "flow", specified_parameters)

    # Assign coupling permeability
    for _, d in gb.edges():
        mg = d["mortar_grid"]
        kn = 2 * kf * np.ones(mg.num_cells) / a
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
