import numpy as np
import scipy.sparse as sps
import porepy as pp

import examples.papers.dfn_transport.discretization as discr
import data

def main():

    input_folder = "../geometries/"
    file_name = input_folder + "example1.fab"
    file_inters = input_folder + "example1.dat"

    out_folder = "solution"

    mesh_size = np.power(2., -4)
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}
    tol = 1e-8
    gb = pp.importer.dfn_3d_from_fab(file_name, file_inters, tol=tol, **mesh_kwargs)

    gb.remove_nodes(lambda g: g.dim == 0)
    gb.compute_geometry()
    gb.assign_node_ordering()

    domain = gb.bounding_box(as_dict=True)

    # setup the flow problem
    physics_flow = "flow"
    key_pressure = "pressure"
    key_discharge = "discharge"

    keys_flow = discr.flow(gb, physics_flow)
    param_flow = {"domain": domain, "k": 1, "physics": physics_flow}
    data.flow(gb, param_flow, tol)

    # solution of the darcy problem
    assembler = pp.Assembler()

    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    up = sps.linalg.spsolve(A, b)

    assembler.distribute_variable(gb, up, block_dof, full_dof)
    for g, d in gb:
        key, key_discr = keys_flow
        flow_discr = d["discretization"][key][key_discr]
        d[key_pressure] = flow_discr.extract_pressure(g, d[key])
        d[key_discharge] = flow_discr.extract_flux(g, d[key])

    # setup the advection-diffusion problem
    physics_advdiff = "transport"

    key_advdiff = discr.advdiff(gb, physics_advdiff)
    param_advdiff = {"domain": domain, "diff": 1e-2, "physics": physics_advdiff,
                     "flux": keys_flow[1]}
    data.advdiff(gb, param_advdiff, tol)

    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    x = sps.linalg.spsolve(A, b)

    assembler.distribute_variable(gb, x, block_dof, full_dof)

    save = pp.Exporter(gb, "solution", folder=out_folder)
    save.write_vtk([key_pressure, key_advdiff])

if __name__ == "__main__":
    main()
