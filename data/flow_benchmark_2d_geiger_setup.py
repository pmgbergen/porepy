#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains utility functions for setting up the problems related to a 2d benhcmark
as described by Flemisch et al (2018).
"""

import numpy as np

import porepy as pp


def add_data(mdg, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    for key in ["param", "is_tangential"]:
        for sd in mdg.subdomains():
            data = mdg._subdomain_data[sd]
            if key not in data:
                data[key] = None
    tol = 1e-5
    a = 1e-4

    for sd, d in mdg.subdomains(return_data=True):
        # Assign aperture
        a_dim = np.power(a, mdg.dim_max() - sd.dim)
        specific_volume = np.ones(sd.num_cells) * a_dim

        # Effective permeability, scaled with aperture.
        kxx = np.ones(sd.num_cells) * np.power(kf, sd.dim < mdg.dim_max()) * specific_volume
        if sd.dim == 2:
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=1, kzz=1)

        compressibility = 1e-10 * specific_volume
        specified_parameters = {
            "second_order_tensor": perm,
            "mass_weight": compressibility,
           }
        # Boundaries
        bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = sd.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[right] = "dir"

            bc_val = np.zeros(sd.num_faces)
            bc_val[bound_faces[left]] = -a_dim * sd.face_areas[bound_faces[left]]
            bc_val[bound_faces[right]] = 1

            bound = pp.BoundaryCondition(sd, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
        else:
            bound = pp.BoundaryCondition(sd, np.empty(0), np.empty(0))
            specified_parameters.update({"bc": bound})

        d["is_tangential"] = True
        pp.initialize_default_data(sd, d, "flow", specified_parameters)

    # Assign coupling permeability
    for intf, d in mdg.interfaces(return_data=True):
        kn = 2 * kf * np.ones(intf.num_cells) / a
        d[pp.PARAMETERS] = pp.Parameters(intf, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

        