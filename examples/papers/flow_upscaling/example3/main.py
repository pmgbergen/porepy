import numpy as np
import scipy.sparse as sps
import porepy as pp

from examples.papers.flow_upscaling.import_grid import grid, square_grid, raw_from_csv

from mpfa_upscalig import MpfaUpscaling

def global_data(g, **kwargs):
    km = 1.
    physics = kwargs.get("physics", "flow")

    param = pp.Parameters(g)
    # set the permeability
    perm = pp.SecondOrderTensor(3, km*np.ones(g.num_cells))
    param.set_tensor("flow", perm)

    # Boundaries
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    if b_faces.size:

        labels = np.array(["dir"] * b_faces.size)
        param.set_bc(physics, pp.BoundaryCondition(g, b_faces, labels))

        bc_val = np.zeros(g.num_faces)
        param.set_bc_val(physics, bc_val)
    else:
        param.set_bc(physics, pp.BoundaryCondition(g, empty, empty))

    return param

def local_node_data(g, d, gb, **kwargs):
    kf = 1e4
    km = 1.
    aperture = 1e-3
    physics = kwargs.get("physics", "flow")

    param = pp.Parameters(g)
    # set the permeability
    if g.dim == 2:
        kxx = km
    else: #g.dim == 1:
        kxx = kf
    perm = pp.SecondOrderTensor(3, kxx*np.ones(g.num_cells))
    param.set_tensor(physics, perm)

    # Assign apertures
    param.set_aperture(np.power(aperture, 2-g.dim))

    return {"param": param}

def local_edge_data(e, d, gb, **kwargs):
    kf = 1e4

    g_l = gb.nodes_of_edge(e)[0]
    mg = d["mortar_grid"]
    check_P = mg.slave_to_mortar_avg()

    aperture = gb.node_props(g_l, "param").get_aperture()
    gamma = check_P * np.power(aperture, 1./(2.-g_l.dim))
    return {"kn": kf * np.ones(mg.num_cells) / gamma}

def compute_discharge(gb, **kwargs):

    physics = kwargs.get("physics", "flow")
    key = kwargs.get("key", "pressure")

    flux = "flux"
    flux_mortar = "lambda_" + flux

    flux_discr = pp.Mpfa(physics)
    flux_coupling = pp.RobinCoupling(key, flux_discr)

    # define the dof and discretization for the grids
    for g, d in gb:
        d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        d[pp.keywords.DISCRETIZATION] = {key: {flux: flux_discr}}

    # define the interface terms to couple the grids
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.keywords.PRIMARY_VARIABLES] = {flux_mortar: {"cells": 1}}
        d[pp.keywords.COUPLING_DISCRETIZATION] = {
            flux: {
                g_slave:  (key, flux),
                g_master: (key, flux),
                e: (flux_mortar, flux_coupling)
            }
        }

    # solution of the problem
    assembler = pp.Assembler()

    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    p = sps.linalg.spsolve(A, b)

    # compute the discharge and fix the sign with f
    assembler.distribute_variable(gb, p, block_dof, full_dof)
    pp.fvutils.compute_discharges(gb, physics=physics, p_name=key, lam_name=flux_mortar)

#    for g, d in gb:
#        if g.dim == 2:
#            print(d["pressure"].size, g.num_cells)
#            pp.plot_grid(g, d["pressure"])

    # extract the discharge and return it
    return gb.node_props(gb.grids_of_dimension(gb.dim_max())[0], "discharge")

if __name__ == "__main__":
    file_geo = "network.csv"
    #file_geo = "../example2/algeroyna_1to10.csv"
    folder = "solution"
    tol = 1e-3
    tol_small = 1e-5

    mesh_args = {'mesh_size_frac': 0.25, "tol": tol}

    g = pp.StructuredTriangleGrid([3, 3], [1, 1])
    g.compute_geometry()

    gb, domain = square_grid(mesh_args)
    g = gb.grids_of_dimension(2)[0]
    pp.plot_grid(g, alpha=0, info="fc")

    np.set_printoptions(linewidth=9999)

    # read the background fractures
    fracs_pts, fracs_edges = pp.importer.lines_from_csv(file_geo)
    #fracs_pts, fracs_edges, _ = raw_from_csv(file_geo, mesh_args)
    #fracs_pts /= np.linalg.norm(np.amax(fracs_pts, axis=1))

    # the data for the local problem
    data_local = {
        "node_data": local_node_data,
        "edge_data": local_edge_data,
        "tol": tol,
        "fractures": {"points": fracs_pts, "edges": fracs_edges},
        "mesh_args": {'mesh_size_frac': 1},
        "compute_discharge": compute_discharge
        }

    data = {
        "param": global_data(g),
        "tol": tol,
        "local_problem": data_local
    }


    #discr = pp.Mpfa("flow")
    #discr.assemble_matrix_rhs(g, data)
    #print(data["flow_flux"])

    discr = MpfaUpscaling("flow")
    discr.assemble_matrix_rhs(g, data)
