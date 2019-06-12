import numpy as np
import os
import porepy as pp


def grid_export(gb, P0_flux, folder):

    if not os.path.exists(folder):
        os.makedirs(folder)

    # export the grids
    for g, d in gb:

        # only 2d fractures
        if g.dim != 2:
            continue

        # extract the id of the fracture
        frac_num = int(d["frac_num"][0])

        # extract the cell faces in I, J format
        cell_faces_I = []
        cell_faces_J = []
        # loop on all the cells and get the faces (from the indices map)
        indices = g.cell_faces.indices
        indptr = g.cell_faces.indptr
        face_cells = -np.ones((g.num_faces, 2), dtype=np.int)
        for cell in np.arange(g.num_cells):
            faces = indices[indptr[cell] : indptr[cell + 1]]
            cell_faces_I.append([cell] * faces.size)
            cell_faces_J.append(faces.tolist())
            for face in faces:
                idx = np.where(face_cells[face, :] == -1)[0]
                face_cells[face, idx[0]] = cell

        # convert to numpy array
        cell_faces_I = np.hstack(cell_faces_I).astype(np.int)
        cell_faces_J = np.hstack(cell_faces_J).astype(np.int)

        # save to file the map
        fname = "g_" + str(frac_num) + "_cell_faces.txt"
        cell_faces = np.vstack((cell_faces_I, cell_faces_J)).T
        np.savetxt(folder + fname, cell_faces, fmt="%d", delimiter=",")

        # save to file the map
        fname = "g_" + str(frac_num) + "_face_cells.txt"
        np.savetxt(folder + fname, face_cells, fmt="%d", delimiter=",")

        # extract the face nodes in I, J format
        face_nodes_I = []
        face_nodes_J = []
        # loop on all the faces and get the nodes (from the indices map)
        indices = g.face_nodes.indices
        indptr = g.face_nodes.indptr
        for face in np.arange(g.num_faces):
            nodes = indices[indptr[face] : indptr[face + 1]]
            face_nodes_I.append([face] * nodes.size)
            face_nodes_J.append(nodes.tolist())

        # convert to numpy array
        face_nodes_I = np.hstack(face_nodes_I)
        face_nodes_J = np.hstack(face_nodes_J)

        # save to file the map
        fname = "g_" + str(frac_num) + "_face_nodes.txt"
        face_nodes = np.vstack((face_nodes_I, face_nodes_J)).T
        np.savetxt(folder + fname, face_nodes, fmt="%d", delimiter=",")

        # save to file the points
        fname = "g_" + str(frac_num) + "_nodes.txt"
        np.savetxt(folder + fname, g.nodes.T, fmt="%10.14f", delimiter=",")

        # save the cell neighbors
        cell_cell_map = np.zeros((g.num_cells, 3))
        cell_connection = g.cell_connection_map()
        indices = cell_connection.indices
        indptr = cell_connection.indptr
        # NOTE: I'm assuming a simplicial grid
        for cell in np.arange(g.num_cells):
            cells = np.sort(indices[indptr[cell] : indptr[cell + 1]])
            # do not save the current cell
            cells = np.setdiff1d(cells, cell, assume_unique=True)
            # in case of boundary cell put -1 as flag
            if cells.size < 3:
                cells = np.append(cells, [-1] * (3 - cells.size))
            cell_cell_map[cell, :] = cells.copy()

        fname = "g_" + str(frac_num) + "_cell_cells.txt"
        np.savetxt(folder + fname, cell_cell_map, fmt="%d", delimiter=",")

        # save to file the cell centroids and volume
        fname = "g_" + str(frac_num) + "_cell_data.txt"
        cell_data = np.vstack((g.cell_volumes, g.cell_centers)).T
        np.savetxt(folder + fname, cell_data, fmt="%10.14f", delimiter=",")

        # save to file the bc type
        fname = "g_" + str(frac_num) + "_face_data.txt"
        bc = d[pp.PARAMETERS]["flow_data"]["bc"]
        bc_tag = bc.is_dir.astype(np.int) + 2 * bc.is_neu.astype(np.int)
        bc_tags = np.vstack((bc_tag, g.tags["bc_flow_id"])).T
        np.savetxt(folder + fname, bc_tags, fmt="%d", delimiter=",")

    # export the connectivity maps
    for g, d in gb:

        # only 1d grid
        if g.dim != 1:
            continue

        # we assume only two intersecting fractures
        faces = np.empty((g.num_cells, 4), dtype=np.int)
        frac_num = np.empty(2, dtype=np.int)

        frac_id = 0
        for e, d_e in gb.edges_of_node(g):
            g_h = gb.nodes_of_edge(e)[1]
            frac_num[frac_id] = gb.node_props(g_h, "frac_num")[0]

            face_cells = d_e["face_cells"].astype(np.int).tocsr()
            # loop on all the cells and get the faces (from the indices map)
            for cell in np.arange(g.num_cells):
                indices = face_cells.indices
                indptr = face_cells.indptr
                faces[cell, (2 * frac_id) : (2 * frac_id + 2)] = indices[
                    indptr[cell] : indptr[cell + 1]
                ]

            frac_id += 1

        faces_sorted = np.empty((g.num_cells, 4), dtype=np.int)
        sort = np.argsort(frac_num)
        for idx, idx_sorted in enumerate(sort):
            faces_sorted[:, (2 * idx) : (2 * idx + 2)] = faces[
                :, (2 * idx_sorted) : (2 * idx_sorted + 2)
            ]

        # save to file
        trace_id = "_".join([str(f) for f in frac_num[sort]])
        fname = "t_" + trace_id + "_faces.txt"
        np.savetxt(folder + fname, faces_sorted, fmt="%d", delimiter=",")

    # export the flux
    if P0_flux is not None:
        for g, d in gb:

            # only 2d grid
            if g.dim != 2:
                continue

            # extract the id of the fracture
            frac_num = int(d["frac_num"][0])

            # save to file the points
            fname = "g_" + str(frac_num) + "_P0_flux.txt"
            np.savetxt(folder + fname, d[P0_flux].T, fmt="%10.14f", delimiter=",")
