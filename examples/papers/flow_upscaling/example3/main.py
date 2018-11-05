import numpy as np
import scipy.sparse as sps
import porepy as pp

from examples.papers.flow_upscaling.import_grid import grid, square_grid, raw_from_csv

def slice_out(f, m):
    return m.indices[m.indptr[f]: m.indptr[f+1]]

def set_data(gb, data):
    for g, d in gb:
        param = pp.Parameters(g)

        # set the permeability
        if g.dim == 2:
            kxx = data["km"]
        else: #g.dim == 1:
            kxx = data["kf"]

        perm = pp.SecondOrderTensor(3, kxx*np.ones(g.num_cells))
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(data['aperture'], 2-g.dim)
        param.set_aperture(aperture*np.ones(g.num_cells))

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:
            labels = ["dir"] * b_faces.size
            param.set_bc("flow", pp.BoundaryCondition(g, b_faces, labels))
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()

        aperture = gb.node_props(g_l, "param").get_aperture()
        gamma = check_P * np.power(aperture, 1./(2.-g.dim))
        d["kn"] = data["kf"] * np.ones(mg.num_cells) / gamma


def set_boundary_conditions(gb, pts, pt, tol):

    bd = np.array([[pt, (pt+1)%pts.shape[1], (pt+2)%pts.shape[1]],
                   [pt, (pt-1)%pts.shape[1], (pt-2)%pts.shape[1]]])

    dist = lambda i, j: np.linalg.norm(pts[:, bd[j, i]] - pts[:, bd[j, i+1]])
    length = np.array([[dist(i, j) for i in np.arange(2)] for j in np.arange(2)])
    sum_length = np.sum(length, axis=1)

    for g, d in gb:
        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:
            b_face_centers = g.face_centers[:, b_faces]
            bc_val = np.zeros(g.num_faces)
            for j in np.arange(2):
                x_0 = np.array([0, length[j, 0]])
                x_1 = np.array([length[j, 0], sum_length[j]])
                y_0 = np.array([1, length[j, 0]/sum_length[j]])
                y_1 = np.array([length[j, 0]/sum_length[j], 0])

                for i in np.arange(2):
                    start = pts[:, bd[j, i]]
                    end = pts[:, bd[j, i+1]]

                    # detect all the points aligned with the segment
                    dist, _ = pp.cg.dist_points_segments(b_face_centers, start, end)
                    mask = np.where(np.logical_and(dist < tol, dist >=-tol))[0]

                    # compute the distance
                    delta = np.tile(start, (mask.size, 1)).T - b_face_centers[:, mask]

                    # define the boundary conditions
                    val = (y_1[i] - y_0[i])/(x_1[i] - x_0[i])
                    bc_val[b_faces[mask]] = np.linalg.norm(delta, axis=0)*val + y_0[i]

            d["param"].set_bc_val("flow", bc_val)


if __name__ == "__main__":
    #file_geo = "network.csv"
    file_geo = "../example2/algeroyna_1to10.csv"
    folder = "solution"
    tol = 1e-3
    tol_small = 1e-5

    mesh_args = {'mesh_size_frac': 1, "tol": tol}
    mesh_args_loc = {'mesh_size_frac': 0.125}

    # define the physical data
    aperture = 1e-2
    data = {
        "aperture": aperture,
        "kf": 1e4,
        "km": 1,
        "tol": tol
    }

    gb, domain = square_grid(mesh_args)
    #pp.plot_grid(gb, alpha=0, info="all")

    np.set_printoptions(linewidth=9999)

    # read the background fractures
    #fracs_pts, fracs_edges = pp.importer.lines_from_csv(file_geo)
    fracs_pts, fracs_edges, _ = raw_from_csv(file_geo, mesh_args)
    fracs_pts /= np.linalg.norm(np.amax(fracs_pts, axis=1))

    solver = pp.MpfaMixedDim("flow")
    for g, d in gb:
        # assuming that the grid max is 2
        if g.dim == 2:

            face_faces = g.face_nodes.T*g.face_nodes
            face_cells = g.face_nodes.T*g.cell_nodes()
            b_nodes = g.get_all_boundary_nodes()

            for f in g.get_internal_faces():
                # extract the faces (removing self), cells, and nodes
                f_loc = np.setdiff1d(slice_out(f, face_faces), f)
                c_loc = slice_out(f, face_cells)
                n_loc = slice_out(f, g.face_nodes)
                f_nodes = g.nodes[:, n_loc]

                # to make the grid conforming consider the current edge as "fracture"
                subdom = {"points": f_nodes[:2, :], "edges": np.array([0, 1]).reshape((2, 1))}
                # keep the face nodes if are on the boundary
                n_loc = n_loc[np.isin(n_loc, b_nodes, assume_unique=True)]
                # collect from: faces, cells, node boundary (if)
                pts = np.hstack((g.cell_centers[:, c_loc], g.face_centers[:, f_loc], g.nodes[:, n_loc]))
                is_cell_centers = np.hstack(([True]*c_loc.size, [False]*(f_loc.size+n_loc.size)))

                # sort the nodes by assuming star shaped region
                mask = pp.utils.sort_points.sort_point_plane(pts, g.face_centers[:, f])
                pts = pts[:, mask]
                is_cell_centers = is_cell_centers[mask]

                # consider the background fractures only limited to the current region
                fracs_pts_int, fracs_edges_int = pp.cg.intersect_polygon_lines(pts, fracs_pts, fracs_edges)
                fracs = {"points": fracs_pts_int, "edges": fracs_edges_int}
                # prepare the data for gmsh
                gb_loc = pp.fracs.meshing.simplex_grid(fracs, domain=pts[:2, :], subdomains=subdom, **mesh_args_loc)
                #pp.plot_grid(gb_loc, alpha=0)

                # save the faces of the face f
                g_h = gb_loc.grids_of_dimension(2)[0]

                dist, _ = pp.cg.dist_points_segments(g_h.face_centers, f_nodes[:, 0], f_nodes[:, 1])
                f_loc = np.where(np.logical_and(dist < tol_small, dist >=-tol_small))[0]
                # compute the sign with respect to f and normalize the result
                sign = np.sign(np.einsum('ij,i->j', g_h.face_normals[:, f_loc], g.face_normals[:, f]))

                # assign common data
                set_data(gb_loc, data)

                # need to compute the upscaled transmissibility
                # we loop on all the cell_centers and give the data
                for pt in np.where(is_cell_centers)[0]:
                    # apply the boundary conditions for the current problem
                    set_boundary_conditions(gb_loc, pts, pt, tol)

                    # Discretize and solve the linear system
                    p = sps.linalg.spsolve(*solver.matrix_rhs(gb_loc))

                    # compute the discharge and fix the sign with f
                    solver.split(gb_loc, "pressure", p)
                    pp.fvutils.compute_discharges(gb_loc)
                    discharge = sign * gb_loc.node_props(g_h, "discharge")[f_loc]
                    print(np.sum(discharge))
                    d
