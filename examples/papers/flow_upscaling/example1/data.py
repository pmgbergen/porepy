import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def flow(gb, model, data):
    tol = data["tol"]

    model_data = model + "_data"
    mu = data["fluid"].dynamic_viscosity()

    for g, d in gb:
        param = {}
        d["is_tangential"] = True

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        # set the permeability
        if g.dim == 2:
            kxx = data['km'] / mu * unity
            perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)
        else: #g.dim == 1:
            kxx = data['kf'] / mu * unity
            perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
        param["second_order_tensor"] = perm

        # Assign apertures
        param["aperture"] = np.power(data['aperture'], 2-g.dim) * unity

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        if b_faces.size:
            (top, bottom, left, right), boundary = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[left + right] = "dir"
            param["bc"] = pp.BoundaryCondition(g, b_faces, labels)

            bc_val[b_faces[right]] = data["bc_flow"]
        else:
            param["bc"] = pp.BoundaryCondition(g, empty, empty)

        param["bc_values"] = bc_val

        d[pp.PARAMETERS] = pp.Parameters(g, model_data, param)
        d[pp.DISCRETIZATION_MATRICES] = {model_data: {}}

    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        aperture = gb.node_props(g_l, pp.PARAMETERS)[model_data]["aperture"]
        gamma = check_P * np.power(aperture, 1/(2.-g.dim))
        kn = data["kf"] / mu * np.ones(mg.num_cells) / gamma

        param = {"normal_diffusivity": kn}

        d[pp.PARAMETERS] = pp.Parameters(e, model_data, param)
        d[pp.DISCRETIZATION_MATRICES] = {model_data: {}}

    return model_data

# ------------------------------------------------------------------------------#

def advdiff(gb, model, model_flow, data):
    tol = data["tol"]

    model_data_adv = model + "_data_adv"
    model_data_diff = model + "_data_diff"

    cm = data["rock"].specific_heat_capacity()
    cw = data["fluid"].specific_heat_capacity()

    rhom = data["rock"].DENSITY
    rhow = data["fluid"].density()

    lm = data["rock"].thermal_conductivity()
    lw = data["fluid"].thermal_conductivity()

    flux_discharge_name = data["flux"]
    flux_mortar_name = data["mortar_flux"]

    for g, d in gb:
        param_adv = {}
        param_diff = {}

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        # Assign the porosity
        if g.dim == 2:
            phi = data["rock"].POROSITY
        else:
            phi = data["porosity_f"]

        ce = phi * rhow * cw + (1-phi) * rhom * cm
        param_adv["mass_weight"] = ce

        # Assign the diffusivity
        l = np.power(lw, phi)*np.power(lm, 1-phi) * unity
        param_diff["second_order_tensor"] = pp.SecondOrderTensor(3, l)

        # Assign apertures
        param_diff["aperture"] = d[pp.PARAMETERS][model_flow]["aperture"]
        param_adv["aperture"] = d[pp.PARAMETERS][model_flow]["aperture"]

        # Flux
        param_adv[flux_discharge_name] = rhow * cw * d[flux_discharge_name]

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        if b_faces.size:
            (top, bottom, left, right), boundary = bc_flag(g, data["domain"], tol)

            labels_adv = np.array(["neu"] * b_faces.size)
            labels_adv[left + right] = ["dir"]

            labels_diff = np.array(["neu"] * b_faces.size)
            labels_diff[right] = ["dir"]

            param_adv["bc"] = pp.BoundaryCondition(g, b_faces, labels_adv)
            param_diff["bc"] = pp.BoundaryCondition(g, b_faces, labels_diff)

            bc_val[b_faces[right]] = data["bc_advdiff"]
        else:
            param_adv["bc"] = pp.BoundaryCondition(g, np.empty(0), np.empty(0))
            param_diff["bc"] = pp.BoundaryCondition(g, np.empty(0), np.empty(0))

        param_adv["bc_values"] = bc_val
        param_diff["bc_values"] = bc_val

        # Assign time step
        param_adv["time_step"] = data["time_step"]

        param = pp.Parameters(g, [model_data_adv, model_data_diff], [param_adv, param_diff])
        d[pp.PARAMETERS] = param
        d[pp.DISCRETIZATION_MATRICES] = {model_data_adv: {}, model_data_diff: {}}

    # Assign coupling discharge and diffusivity
    for e, d in gb.edges():
        param_adv = {}
        param_diff = {}

        param_adv[flux_discharge_name] = d[flux_mortar_name] * rhow * cw

        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        aperture = gb.node_props(g_l, pp.PARAMETERS)[model_data_diff]["aperture"]
        gamma = np.power(aperture, 1/(2.-g.dim))

        k = gb.node_props(g_l, pp.PARAMETERS)[model_data_diff]["second_order_tensor"]

        param_diff["normal_diffusivity"] = check_P.dot(k.values[0, 0, :] / gamma)

        param = pp.Parameters(e, [model_data_adv, model_data_diff], [param_adv, param_diff])
        d[pp.PARAMETERS] = param
        d[pp.DISCRETIZATION_MATRICES] = {model_data_adv: {}, model_data_diff: {}}

    return model_data_adv, model_data_diff

# ------------------------------------------------------------------------------#

def bc_flag(g, domain, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    top = b_face_centers[1] > domain["ymax"] - tol
    bottom = b_face_centers[1] < domain["ymin"] + tol
    left = b_face_centers[0] < domain["xmin"] + tol
    right = b_face_centers[0] > domain["xmax"] - tol
    boundary = top + bottom + left + right

    return (top, bottom, left, right), boundary
