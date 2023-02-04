"""
This file contains utility functions for setting up the problems related to a 2d
benhcmark as described by Flemisch et al (2018).
"""

import numpy as np

import porepy as pp


def add_data(mdg: pp.MixedDimensionalGrid, domain: pp.Domain, kf: float) -> None:
    """
    Define the permeability, apertures, boundary conditions and update
    data of the corresponding mixed-dimensional grid.

    Parameters:
        mdg: mixed-dimensional grid.
        domain: Domain object.
        kf: scalar permeability value.

    """
    tol = 1e-5
    a = 1e-4

    for sd, sd_data in mdg.subdomains(return_data=True):
        # Assign aperture
        a_dim = np.power(a, mdg.dim_max() - sd.dim)
        aperture = np.ones(sd.num_cells) * a_dim

        # Effective permeability, scaled with aperture.
        kxx = np.ones(sd.num_cells) * np.power(kf, sd.dim < mdg.dim_max()) * aperture
        if sd.dim == 2:
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=np.ones(sd.num_cells))
        else:
            perm = pp.SecondOrderTensor(
                kxx=kxx, kyy=np.ones(sd.num_cells), kzz=np.ones(sd.num_cells)
            )

        specified_parameters: dict = {"second_order_tensor": perm}
        # Boundaries
        bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = sd.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain.bounding_box["xmin"] + tol
            right = bound_face_centers[0, :] > domain.bounding_box["xmax"] - tol

            bc_val = np.zeros(sd.num_faces)
            bc_val[bound_faces[left]] = -a_dim * sd.face_areas[bound_faces[left]]
            bc_val[bound_faces[right]] = 1

            bound = pp.BoundaryCondition(sd, bound_faces[right], "dir")
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
        else:
            bound = pp.BoundaryCondition(sd)
            specified_parameters.update({"bc": bound})

        sd_data["is_tangential"] = True
        pp.initialize_default_data(sd, sd_data, "flow", specified_parameters)

    # Assign coupling permeability
    for intf, intf_data in mdg.interfaces(return_data=True):
        kn = 2 * kf * np.ones(intf.num_cells) / a
        intf_data[pp.PARAMETERS] = pp.Parameters(
            intf, ["flow"], [{"normal_diffusivity": kn}]
        )
        intf_data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
