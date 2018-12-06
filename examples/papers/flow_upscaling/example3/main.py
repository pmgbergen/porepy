import numpy as np
import scipy.sparse as sps
import porepy as pp

from examples.papers.flow_upscaling.import_grid import grid, square_grid, raw_from_csv

from mpfa_upscalig import MpfaUpscaling

def global_data(g, keyword="flow"):
    km = 1.

    param = {}

    # set the permeability
    perm = pp.SecondOrderTensor(3, km*np.ones(g.num_cells))
    param["second_order_tensor"] = perm

    # Assign apertures
    param["aperture"] = np.ones(g.num_cells)

    # Boundaries
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bc_val = np.zeros(g.num_faces)
    if b_faces.size:
        labels = np.array(["dir"] * b_faces.size)
        param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
    else:
        param["bc"] = pp.BoundaryCondition(g, np.empty(0), np.empty(0))

    param["bc_values"] = bc_val

    data = {pp.PARAMETERS: pp.Parameters(g, keyword, param),
            pp.DISCRETIZATION_MATRICES: {keyword: {}}}
    return data

def local_node_data(g, d, gb, **kwargs):
    # do not set the boundary conditions,
    # the local grid bucket is different than the global one

    kf = 1e4
    km = 1.
    aperture = 1e-3
    keyword = kwargs.get("keyword", "flow")
    param = {}

    # set the permeability and aperture
    if g.dim == 2:
        kxx = km
    else: #g.dim == 1:
        kxx = kf
    perm = pp.SecondOrderTensor(3, kxx*np.ones(g.num_cells))
    param["second_order_tensor"] = perm

    # Assign apertures
    param["aperture"] = np.power(aperture, 2-g.dim)*np.ones(g.num_cells)

    data = {pp.PARAMETERS: pp.Parameters(g, keyword, param),
            pp.DISCRETIZATION_MATRICES: {keyword: {}}}
    return data

def local_edge_data(e, d, gb, **kwargs):
    kf = 1e4

    g_l = gb.nodes_of_edge(e)[0]
    mg = d["mortar_grid"]
    check_P = mg.slave_to_mortar_avg()

    aperture = gb.node_props(g_l, "param").get_aperture()
    gamma = check_P * np.power(aperture, 1./(2.-g_l.dim))
    return {"kn": kf * np.ones(mg.num_cells) / gamma}

def compute_discharge(gb, **kwargs):

    keyword = kwargs.get("keyword", "flow")
    variable = kwargs.get("variable", "pressure")

    discr_id = "flow"
    mortar = "lambda_" + variable

    discr = pp.Mpfa(keyword)
    coupling = pp.RobinCoupling(keyword, discr)

    # define the dof and discretization for the grids
    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
        d[pp.DISCRETIZATION] = {variable: {discr_id: discr}}

    # define the interface terms to couple the grids
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            flux: {
                g_slave:  (variable, discr_id),
                g_master: (variable, discr_id),
                e: (mortar, coupling)
            }
        }

    # solution of the problem
    assembler = pp.Assembler()

    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    p = sps.linalg.spsolve(A, b)

    # compute the discharge and fix the sign with f
    assembler.distribute_variable(gb, p, block_dof, full_dof)
    flux_name = "darcy_flux"
    pp.fvutils.compute_darcy_flux(gb, keyword, flux_name, variable, mortar)

#    for g, d in gb:
#        if g.dim == 2:
#            print(d["pressure"].size, g.num_cells)
#            pp.plot_grid(g, d["pressure"])

    # extract the discharge and return it
    return gb.node_props(gb.grids_of_dimension(gb.dim_max())[0], flux_name)

if __name__ == "__main__":
    file_geo = "network.csv"
    #file_geo = "../example2/algeroyna_1to10.csv"
    folder = "solution"
    tol = 1e-3
    tol_small = 1e-5

    mesh_args = {'mesh_size_frac': 0.25, "tol": tol}

    g = pp.StructuredTriangleGrid([2, 2], [1, 1])
    g.compute_geometry()

    #gb, domain = square_grid(mesh_args)
    #g = gb.grids_of_dimension(2)[0]
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

    data = global_data(g)
    data.update({"tol": tol, "local_problem": data_local})

    #discr = pp.Mpfa("flow")
    #discr.assemble_matrix_rhs(g, data)
    #print(data["flow_flux"])

    discr = MpfaUpscaling("flow")
    discr.assemble_matrix_rhs(g, data)
