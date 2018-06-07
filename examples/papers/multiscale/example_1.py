import numpy as np
import scipy.sparse as sps
import porepy as pp

#------------------------------------------------------------------------------#

def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(['param'])
    tol = 1e-5
    a = 1e-2

    for g, d in gb:
        param = pp.Parameters(g)

        # Permeability
        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        if g.dim == 2:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        # Boundaries
        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain['xmin'] + tol
            right = bound_face_centers[0, :] > domain['xmax'] - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(right, left)] = 'dir'

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = 0
            bc_val[bound_faces[right]] = 1

            param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(
                g, np.empty(0), np.empty(0)))

        d['param'] = param

    # Assign coupling permeability
    gb.add_edge_props('kn')
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d['mortar_grid']
        check_P = mg.low_to_mortar_avg()

        gamma = check_P*gb.node_props(g_l, 'param').get_aperture()
        d['kn'] = kf*np.ones(mg.num_cells) / gamma

#------------------------------------------------------------------------------#

def write_network(file_name):
    network = "FID,START_X,START_Y,END_X,END_Y\n"
    network += "0,0.5,0,0.5,1\n"
    network += "1,0.25,0.5,1,0.5"
    with open(file_name, "w") as text_file:
        text_file.write(network)

#------------------------------------------------------------------------------#

def make_grid_bucket(mesh_size, is_coarse=False):
    mesh_kwargs = {}
    mesh_kwargs = {'mesh_size_frac': mesh_size,
                   'mesh_size_min': mesh_size / 20}


    domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

    file_name = 'network_geiger.csv'
    write_network(file_name)
    gb = pp.importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain)
    gb.compute_geometry()
    if is_coarse:
        pp.coarsening.coarsen(gb, 'by_volume')
    gb.assign_node_ordering()
    return gb, domain

#------------------------------------------------------------------------------#

def main(kf, is_coarse=False):

    mesh_size = 1#0.045
    gb, domain = make_grid_bucket(mesh_size)

    # Assign parameters
    add_data(gb, domain, kf)

    # Choose and define the solvers and coupler
    solver_flow = pp.DualVEMMixedDim('flow')
    A, b = solver_flow.matrix_rhs(gb, return_bmat=True)

    # Select the grid of higher dimension
    g_h = gb.grids_of_dimension(gb.dim_max())[0]
    pos_hn = [gb.node_props(g_h, 'node_number')]
    dof_h = A[pos_hn[0], pos_hn[0]].shape[0]

    # Select the grid positions of one dimension down
    pos_ln = [d['node_number'] for g, d in gb if g.dim < g_h.dim]

    # select the positions of the mortars blocks
    num_nodes = gb.num_graph_nodes()
    pos_he = []
    pos_le = []
    for e, d in gb.edges():
        gl_h = gb.nodes_of_edge(e)[1]
        if gl_h.dim == g_h.dim:
            pos_he.append(d['edge_number'] + num_nodes)
        else:
            pos_le.append(d['edge_number'] + num_nodes)

    # extract the blocks for the higher dimension
    A_h = sps.bmat(A[np.ix_(pos_hn+pos_he, pos_hn+pos_he)])
    C_h = sps.bmat(A[np.ix_(pos_he, pos_ln)])

    # Select the grid positions of lower dimension
    A_l = sps.bmat(A[np.ix_(pos_ln+pos_le, pos_ln+pos_le)])
    C_l = sps.bmat(A[np.ix_(pos_ln, pos_he)])
    print(C_h.shape)

    # construct the rhs with all the active pressure dof in the 1 co-dimension
    if_p = np.zeros(C_h.shape[1], dtype=np.bool)
    pos = 0
    for g, d in gb:
        if g.dim != g_h.dim:
            pos += g.num_faces
            if g.dim == g_h.dim - 1:
                if_p[pos:pos+g.num_cells] = True
            pos += g.num_cells

    # compute the bases
    num_bases = np.sum(if_p)
    dof_bases = C_h.shape[0]
    print(dof_bases)
    bases = np.zeros((num_bases, C_h.shape[1]))

#    for e, d in gb.edges():
#        g_m = d['mortar_grid']
#        print(g_m.low_to_mortar_avg())
#        print('---')

    LU = sps.linalg.factorized(A_h.tocsc())
    for idx, dof_basis in enumerate(np.where(if_p)[0]):
        rhs = np.zeros(if_p.size)
        rhs[dof_basis] = 1.
        x = LU(np.concatenate((np.zeros(dof_h), -C_h*rhs)))
        bases[idx, if_p] = x[-dof_bases:]

    # construct the problem in the fracture network
    print(bases)


    print(C_l.shape, bases.shape)

#        save = pp.Exporter(g_h, "vem_"+str(idx)+"_", folder='result')
#        save.write_vtk({'pressure': x[g_h.num_faces:dof_h]})

    dss

    solver_source = pp.DualSourceMixedDim('flow')

    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    gb.add_node_props(["discharge", 'pressure', "P0u"])
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.extract_p(gb, "up", 'pressure')
    solver_flow.project_u(gb, "discharge", "P0u")

    save = pp.Exporter(gb, "vem", folder="vem")
    save.write_vtk(['pressure', "P0u"])

#------------------------------------------------------------------------------#

if __name__ == "__main__":
    kf = 1e2
    main(kf)
