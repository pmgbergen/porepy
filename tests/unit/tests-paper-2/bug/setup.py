import numpy as np
import porepy as pp


def gb(
    mesh_size,
    file_name="network.csv",
    domain={"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
):

    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # assign the flag for the low permeable fractures
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 200}
    # Generate a mixed-dimensional mesh
    mdg = network.mesh(mesh_kwargs)
    # coarse the grid
    pp.coarsening.coarsen(mdg, "by_volume")
    # set the flags for the fractures
    set_flag(mdg)
    return mdg


def data(mdg, data_key="flow"):

    # Thickness of fracture
    aperture = 1e-4
    fracture_perm = 1

    for sd, d in mdg.subdomains(return_data=True):
        # The concept of specific volumes accounts for the thickness
        # of the fracture, which is collapsed in the mixed-dimensional
        # model.
        specific_volumes = np.power(aperture, mdg.dim_max() - sd.dim)
        # Permeability
        k = np.ones(sd.num_cells) * specific_volumes
        if sd.dim < mdg.dim_max():
            k *= fracture_perm
        perm = pp.SecondOrderTensor(k)

        # Zero scalar source
        f = 0.0 * np.ones(sd.num_cells) * specific_volumes

        # Boundary conditions
        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, b_faces, ["dir"] * b_faces.size)
        bc_val = np.zeros(sd.num_faces)
        # bc_val[b_faces] = g.face_centers[1, b_faces]

        parameters = {
            "second_order_tensor": perm,
            "source": f,
            "bc": bc,
            "bc_values": bc_val,
        }
        pp.initialize_data(sd, d, data_key, parameters)

    for mg, d in mdg.interfaces(return_data=True):
        # Division through aperture/2 may be thought of as taking the gradient, i.e.
        # dividing by the distance from the matrix to the center of the fracture.
        kn = fracture_perm / (aperture / 2)
        pp.initialize_data(mg, d, data_key, {"normal_diffusivity": kn})


def set_perm(mdg, k_low, k_high, aperture=1e-4):
    # First we set the fracture permeabilities
    def return_perm(is_low):
        if is_low:
            return k_low
        else:
            return k_high

    for sd, d in mdg.subdomains(True):
        if sd.dim < mdg.dim_max():
            fracture_perm = return_perm(d["is_low"])
            specific_volumes = np.power(aperture, mdg.dim_max() - sd.dim)
            k = fracture_perm * np.ones(sd.num_cells) * specific_volumes

            d["parameters"]["flow"]["second_order_tensor"] = pp.SecondOrderTensor(k)

    # Then we set the normal permeabilities
    def return_mortar_perm(is_low, dim):
        if dim == 1:
            return return_perm(is_low)
        elif is_low:
            return 2.0 / (1.0 / k_low + 1.0 / k_high)
        else:
            return k_high

    for mg, d in mdg.interfaces(True):
        dim = mg.dim
        fracture_perm = return_mortar_perm(d["is_low"], dim)
        kn = fracture_perm / (aperture / 2)
        d["parameters"]["flow"]["normal_diffusivity"] = kn


def set_flag(mdg: pp.MixedDimensionalGrid, tol=1e-3):
    # set the key for the low peremable fractures

    for g, d in mdg.subdomains(return_data=True):
        d["is_low"] = False

        if g.dim == 1:
            f_3 = (g.nodes[0, :] - 0.15) / (0.4 - 0.15) - (g.nodes[1, :] - 0.9167) / (
                0.5 - 0.9167
            )
            if np.sum(np.abs(f_3)) < tol:
                d["is_low"] = True

            f_4 = (g.nodes[0, :] - 0.65) / (0.849723 - 0.65) - (
                g.nodes[1, :] - 0.8333
            ) / (0.167625 - 0.8333)
            if np.sum(np.abs(f_4)) < tol:
                d["is_low"] = True

    # we set the flag for the intersections
    for e in mdg.interfaces():
        gl, gh = mdg.interface_to_subdomain_pair(e)

        if gl.dim == 0 and mdg.node_props(gh, "is_low"):
            gl_data = mdg.subdomain_data(gl)
            gl_data["is_low"] = True

    # The flag on the mortar is inherited from the lower-dim grid
    for e, d in mdg.interfaces(return_data=True):
        gl, gh = mdg.interface_to_subdomain_pair(e)
        gl_data = mdg.subdomain_data(gl)
        d["is_low"] = gl_data["is_low"]


# ------------------------------------------------------------------------------#
