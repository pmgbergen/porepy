import numpy as np
import scipy.sparse as sps
import porepy as pp

from examples.papers.flow_upscaling.import_grid import grid, square_grid

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
            if g.dim == 2:
                labels = ["dir"] * b_faces.size
            else:
                labels = ["neu"] * b_faces.size
                print(b_faces)
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
        gamma = check_P * np.power(aperture, 1/(2.-g.dim))
        d["kn"] = data["kf"] * np.ones(mg.num_cells) / gamma


def set_boundary_conditions(gb, pts, pt, pt_others, tol):

    for g, d in gb:
        if g.dim == 2:
            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size != 0:
                b_face_centers = g.face_centers[:, b_faces]
                bc_val = np.zeros(g.num_faces)
                start = pts[:, pt]
                for pt_other in pt_others:
                    end = pts[:, pt_other]

                    # detect all the points aligned with the segment
                    dist, _ = pp.cg.dist_points_segments(b_face_centers, start, end)
                    mask = np.where(np.logical_and(dist < tol, dist >=-tol))[0]

                    # compute the distance
                    delta = b_face_centers[:, mask] - np.tile(start, (mask.size, 1)).T
                    length = np.linalg.norm(start - end)

                    # define the boundary conditions
                    bc_val[b_faces[mask]] = 1 - np.linalg.norm(delta, axis=0)/length

                d["param"].set_bc_val("flow", bc_val)


if __name__ == "__main__":
    file_geo = "network.csv"
    folder = "solution"
    tol = 1e-8

    mesh_args = {'mesh_size_frac': 1, "tol": tol}
    mesh_args_loc = {'mesh_size_frac': 0.125}

    # define the physical data
    aperture = 1e-2
    data = {
        "aperture": aperture,
        "kf": 1e2,
        "km": 1,
        "tol": tol
    }

    gb, domain = square_grid(mesh_args)
    #pp.plot_grid(gb, alpha=0, info="all")

    np.set_printoptions(linewidth=9999)

    # read the background fractures
    fracs_pts, fracs_edges = pp.importer.lines_from_csv(file_geo)

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
                pts = np.hstack((g.face_centers[:, f_loc], g.cell_centers[:, c_loc], g.nodes[:, n_loc]))

                # sort the nodes by assuming star shaped region
                mask = pp.utils.sort_points.sort_point_plane(pts, g.face_centers[:, f])
                pts = pts[:, mask]

                # LE FRATTURE DI BACKGROUND DEVONO ESSERE TAGLIATE ALTRIMENTI ROMPONO
                fracs_pts_int, fracs_edges_int = pp.cg.intersect_polygon_lines(pts, fracs_pts, fracs_edges)
                fracs = {"points": fracs_pts_int, "edges": fracs_edges_int}
                # prepare the data for gmsh
                gb_loc = pp.fracs.meshing.simplex_grid(fracs, domain=pts[:2, :], subdomains=subdom, **mesh_args_loc)
                pp.plot_grid(gb_loc, alpha=0, info="f")

                # save the faces of the face f
                g_h = gb_loc.grids_of_dimension(2)[0]
                faces_loc_f = g_h.get_internal_faces()

                #import pdb; pdb.set_trace()
                dist, _ = pp.cg.dist_points_segments(g_h.face_centers[:, faces_loc_f],
                                                     f_nodes[:, 0], f_nodes[:, 1])
                mask = np.where(np.logical_and(dist < tol, dist >=-tol))[0]
                faces_loc_f = faces_loc_f[mask]

                # assign common data
                set_data(gb_loc, data)

                # need to compute the upscaled transmissibility
                # we loop on all the pts and give the data
                for pt in np.arange(pts.shape[1]):
                    pt_others = [(pt+1)%pts.shape[1], (pt-1)%pts.shape[1]]

                    # apply the boundary conditions for the current problem
                    set_boundary_conditions(gb_loc, pts, pt, pt_others, tol)

                    # Discretize and solve the linear system
                    p = sps.linalg.spsolve(*solver.matrix_rhs(gb_loc))
                    solver.split(gb_loc, "pressure", p)

                    save = pp.Exporter(gb_loc, "fv", folder="fv")
                    save.write_vtk(["pressure"])
                    s

                    # compute the discharge
                    pp.fvutils.compute_discharges(gb_loc)

                    discharge = gb_loc.node_props(g_h, "discharge")[faces_loc_f]
                    # MI MANCA IL SEGNO O COMUNQUE E; DA CONTROLLARE
                    print(discharge)
                    # Store the solution
                    gb.add_node_props(["pressure"])
                    pp.plot_grid(gb_loc, "pressure")
