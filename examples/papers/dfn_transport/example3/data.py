import numpy as np
import porepy as pp


def flow(gb, data, tol):
    physics = data["physics"]

    for g, d in gb:

        unity = np.ones(g.num_cells)
        empty = np.empty(0)

        d["frac_num"] = (g.frac_num if g.dim == 2 else -1) * unity
        d["cell_volumes"] = g.cell_volumes
        d["is_tangential"] = True
        d["tol"] = tol

        param = pp.Parameters(g)

        kxx = data["k"] * unity
        perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)
        param.set_tensor(physics, perm)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size:
            in_flow, out_flow = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[in_flow + out_flow] = "dir"
            param.set_bc(physics, pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[in_flow]] = 1
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
            in_flow, out_flow = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[in_flow + out_flow] = ["dir"]
            param.set_bc(physics, pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[in_flow]] = 1
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

    # define inflow type boundary conditions
    out_flow_start = np.array([1.011125, 0.249154, 0.598708])
    out_flow_end = np.array([1.012528, 0.190858, 0.886822])

    # detect all the points aligned with the segment
    dist, _ = pp.cg.dist_points_segments(b_face_centers, out_flow_start, out_flow_end)
    dist = dist.flatten()
    out_flow = np.logical_and(dist < tol, dist >= -tol)

    # define outflow type boundary conditions
    in_flow_start = np.array([0.206507, 0.896131, 0.183632])
    in_flow_end = np.array([0.181980, 0.813947, 0.478618])

    # detect all the points aligned with the segment
    dist, _ = pp.cg.dist_points_segments(b_face_centers, in_flow_start, in_flow_end)
    dist = dist.flatten()
    in_flow = np.logical_and(dist < tol, dist >= -tol)

    return in_flow, out_flow
