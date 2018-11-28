import numpy as np
import porepy as pp

def flow(gb, data, tol):
    physics = data["physics"]

    for g, d in gb:
        param = pp.Parameters(g)
        d["is_tangential"] = True

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        kxx = data["k"] * unity
        perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)
        param.set_tensor(physics, perm)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size:
            (top, bottom, left, right), boundary = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[boundary] = "dir"
            param.set_bc(physics, pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[right]] = 1
            param.set_bc_val(physics, bc_val)
        else:
            param.set_bc(physics, pp.BoundaryCondition(g, empty, empty))

        d["param"] = param

# ------------------------------------------------------------------------------#

def advdiff(gb, data, tol):
    physics = data["physics"]

    for g, d in gb:
        param = d["param"]
        d["deltaT"] = data["deltaT"]

        kxx = data["diff"] * np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx)
        param.set_tensor(physics, perm)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size:
            (top, bottom, left, right), boundary = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[boundary] = ["dir"]
            param.set_bc(physics, pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[right]] = 1
            param.set_bc_val(physics, bc_val)
        else:
            param.set_bc(physics, pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

    lambda_flux = "lambda_" + data["flux"]
    for _, d in gb.edges():
        d["flux_field"] = d[lambda_flux]

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
