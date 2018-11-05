"""
Example illustrating the elimination of intersection cells for transport problem.
Note that with the current parameters and time steps, the non-eliminated version
is unstable. This happens because the CFL method of upwind.py does not take 0d grids
into account, but illustrates the power of the elimination neatly.

The example is based on the test_coupler.py from transport/examples.
"""
import numpy as np
import porepy as pp

from porepy.viz.exporter import export_vtk, export_pvd

from porepy.numerics.mixed_dim import coupler, condensation

# ------------------------------------------------------------------------------#


def add_data_transport(gb):

    gb.add_node_props(["bc", "bc_val", "discharge", "apertures"])
    for g, d in gb:
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        index = np.nonzero(
            abs(g.face_centers[:, b_faces]) == np.ones([3, b_faces.size])
        )[1]
        b_faces = b_faces[index]
        d["bc"] = pp.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        d["bc_val"] = {"dir": np.ones(b_faces.size)}
        d["apertures"] = np.ones(g.num_cells) * np.power(.01, float(g.dim < 3))
        d["discharge"] = pp.Upwind().discharge(
            g, [1, 1, 2 * np.power(1000, g.dim < 2)], d["apertures"]
        )

    gb.add_edge_prop("discharge")
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        d["discharge"] = gb.node_prop(g_h, "discharge")


# ------------------------------------------------------------------------------#


if __name__ == "__main__":
    np.set_printoptions(linewidth=999999)
    np.set_printoptions(precision=7)

    f_1 = np.array([[-.8, .8, .8, -.8], [0, 0, 0, 0], [-.8, -.8, .8, .8]])
    f_2 = np.array([[0, 0, 0, 0], [-.8, .8, .8, -.8], [-.8, -.8, .8, .8]])
    f_3 = np.array([[-.8, .8, .8, -.8], [-.8, -.8, .8, .8], [0, 0, 0, 0]])

    f_set = [f_1, f_2, f_3]

    domain = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1, "zmin": -1, "zmax": 1}

    Nx, Ny = 10, 10
    path_to_gmsh = "~/gmsh-2.16.0-Linux/bin/gmsh"

    gb = pp.fracs.meshing.simplex_grid(f_set, domain, gmsh_path=path_to_gmsh)
    gb.assign_node_ordering()

    ################## Transport solver ##################

    print("Compute global matrix and rhs for the advection problem")
    gb_r, elimination_data = gb.duplicate_without_dimension(0)
    condensation.compute_elimination_fluxes(gb, gb_r, elimination_data)

    add_data_transport(gb)
    add_data_transport(gb_r)

    upwind_solver = pp.Upwind()
    upwind_cc = pp.UpwindCoupling(upwind_solver)
    coupler_solver = coupler.Coupler(upwind_solver, upwind_cc)
    U, rhs = coupler_solver.matrix_rhs(gb)
    U_r, rhs_r = coupler_solver.matrix_rhs(gb_r)

    deltaT = np.amin([upwind_solver.cfl(g, d) for g, d in gb])
    deltaT_r = np.amin([upwind_solver.cfl(g, d) for g, d in gb_r])

    T = deltaT * max(Nx, Ny) * 4

    gb.add_node_prop("deltaT", None, deltaT)
    gb_r.add_node_prop("deltaT", None, deltaT_r)

    mass_solver = pp.MassMatrix()
    coupler_solver = coupler.Coupler(mass_solver)
    M, _ = coupler_solver.matrix_rhs(gb)
    M_r, _ = coupler_solver.matrix_rhs(gb_r)

    inv_mass_solver = pp.InvMassMatrix()
    coupler_solver = coupler.Coupler(inv_mass_solver)
    invM, _ = coupler_solver.matrix_rhs(gb)
    invM_r, _ = coupler_solver.matrix_rhs(gb_r)

    totDof = np.sum(gb.nodes_prop(gb.nodes(), "dof"))
    totDof_r = np.sum(gb_r.nodes_prop(gb_r.nodes(), "dof"))
    conc = np.zeros(totDof)
    conc_r = np.zeros(totDof_r)

    M_minus_U = M - U
    M_minus_U_r = M_r - U_r

    # Loop over the time
    file_name = "conc"
    file_name_r = "conc_r"
    i, i_export = 0, 0
    export_every = 2
    step_to_export = np.empty(0)
    print("Start the time loop")
    print(deltaT, deltaT_r)

    while i * deltaT <= T:

        # Update the solution
        print("Solve the current state (", i * deltaT, ",", T, ")")
        conc = invM.dot((M_minus_U).dot(conc) + rhs)
        coupler_solver.split(gb, "conc", conc)

        if i % export_every == 0:
            print("Export the solution of the current state")
            export_vtk(gb, file_name, ["conc"], time_step=i_export, folder="simu")

            step_to_export = np.r_[step_to_export, i]
            i_export += 1

        i += 1

    export_pvd(gb, file_name, step_to_export * deltaT, folder="simu")

    step_to_export = np.empty(0)
    i, i_export = 0, 0
    while i * deltaT_r <= T:

        # Update the solution
        conc_r = invM_r.dot((M_minus_U_r).dot(conc_r) + rhs_r)
        coupler_solver.split(gb_r, "conc_r", conc_r)
        if i % export_every == 0:
            print("Export the solution of the current state")
            export_vtk(gb_r, file_name_r, ["conc_r"], time_step=i_export, folder="simu")
            step_to_export = np.r_[step_to_export, i]
            i_export += 1

        i += 1

    export_pvd(gb_r, file_name_r, step_to_export * deltaT_r, folder="simu")
