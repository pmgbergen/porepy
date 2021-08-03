import numpy as np
import scipy.sparse as sps

import porepy as pp


def read_grid(fn):
    """
    Read a MRST grid from a file.
    See also pp.io.grid_writer.write_grid(..).

    Parameters:
    fn (String): The file name. This will be passed to open() using 'r'

    Returns:
    g  (Grid): The grid read from file
    """
    # Open file and start writing
    with open(fn, "r") as infile:
        # Read grid info
        lines = infile.readlines()
        data = np.array(lines[0].replace("\n", "").split(" "), dtype=int)

        (
            dim,
            num_cells,
            num_faces,
            num_nodes,
            num_face_nodes,
            num_cell_faces,
            has_cell_facetag,
            has_indexmap,
        ) = data

        cartDims = np.array(lines[1].replace("\n", "").split(" ")[1:], dtype=int)

        nodes = np.array(lines[2].replace("\n", "").split(" "), dtype=float)
        nodes = np.reshape(nodes, (dim, -1), "F")
        nodes = np.r_[nodes, np.zeros((3 - dim, nodes.shape[1]))]

        # Face nodes
        face_nodePtr = np.array(lines[3].replace("\n", "").split(" "), dtype=int)

        face_nodesIndices = np.array(lines[4].replace("\n", "").split(" "), dtype=int)

        data = np.ones(face_nodesIndices.size, dtype=bool)
        face_nodes = sps.csc_matrix((data, face_nodesIndices, face_nodePtr))

        # neighbourship
        face_cells = np.array(lines[5].replace("\n", "").split(" "), dtype=int)
        face_cells = np.reshape(face_cells, (2, -1), "C")

        # Face areas
        face_areas = np.array(lines[6].replace("\n", "").split(" "), dtype=float)

        # Face centers
        face_centers = np.array(lines[7].replace("\n", "").split(" "), dtype=float)
        face_centers = np.reshape(face_centers, (dim, -1), "F")
        face_centers = np.r_[face_centers, np.zeros((3 - dim, face_centers.shape[1]))]

        # Face normals
        face_normals = np.array(lines[8].replace("\n", "").split(" "), dtype=float)
        face_normals = np.reshape(face_normals, (dim, -1), "F")
        face_normals = np.r_[face_normals, np.zeros((3 - dim, face_normals.shape[1]))]

        # Cell faces
        cell_facePtr = np.array(lines[9].replace("\n", "").split(" "), dtype=int)

        cell_faces_indices = np.array(lines[10].replace("\n", "").split(" "), dtype=int)

        if cell_faces_indices.size == 2 * num_cell_faces:
            cell_facetags = cell_faces_indices[1::2]
            cell_faces_indices = cell_faces_indices[0::2]
        elif cell_faces_indices.size == num_cell_faces:
            cell_faces = cell_faces_indices.ravel()
        else:
            raise RuntimeError("Could not read cell_face map. Wrong dimensions")
        faces_per_cell = np.diff(cell_facePtr)
        ci = np.repeat(np.arange(num_cells), faces_per_cell)
        sgn = np.ones(num_cell_faces, dtype=int)
        sgn[face_cells[1, cell_faces_indices] == ci] = -1.0
        cell_faces = sps.csc_matrix((sgn, cell_faces_indices, cell_facePtr))

        # index map
        if has_indexmap:
            index_map = np.array(lines[11].replace("\n", "").split(" "), dtype=int)
            index_map = np.array(index_map, dtype=int).ravel()
            line_num = 12
        else:
            line_num = 11

        # Cell volumes
        cell_volumes = np.array(
            lines[line_num].replace("\n", "").split(" "), dtype=float
        )
        line_num += 1
        # Cell centers
        cell_centers = np.array(
            lines[line_num].replace("\n", "").split(" "), dtype=float
        )
        cell_centers = np.reshape(cell_centers, (dim, -1), "F")
        cell_centers = np.r_[cell_centers, np.zeros((3 - dim, cell_centers.shape[1]))]

    g = pp.Grid(dim, nodes, face_nodes, cell_faces, "grid reader")
    if has_indexmap:
        g.cell_index_map = index_map
    if has_cell_facetag:
        g.cell_facetag = cell_facetags
    g.face_cells = face_cells
    g.face_areas = face_areas
    g.face_normals = face_normals
    g.face_centers = face_centers
    g.cell_volumes = cell_volumes
    g.cell_centers = cell_centers
    g.cart_dims = cartDims

    return g


def read_mrst_grid(fn):
    """
    Read a MRST grid from a file.
    See also pp.io.grid_writer.write_grid(..).

    Parameters:
    fn (String): The file name. This will be passed to open() using 'r'

    Returns:
    g  (Grid): The grid read from file
    """
    # Open file and start writing
    with open(fn, "r") as infile:
        # Read grid info
        lines = infile.readlines()
        data = np.array(lines[0].replace("\n", "").split(" ")[1:], dtype=int)

        dim, num_cells, num_faces, num_nodes, num_face_nodes, num_cell_faces = data
        # line 1 contains boolean index for cell_facetag. However, we don't need this
        # as we can infer it directly from the shape of cell_face_indices.
        has_indexmap = bool(lines[2][0])
        l_num = 3

        cartDims = np.array(lines[l_num].replace("\n", "").split(" ")[1:], dtype=int)
        l_num += 1

        nodes = [
            line.replace("\n", "").split(" ")[1:] for line in lines[4 : num_nodes + 4]
        ]
        nodes = np.array(nodes, dtype=float).T
        offset = num_nodes + 4

        # Face nodes
        face_nodePtr = [
            line.replace("\n", "").split(" ")
            for line in lines[offset : num_faces + 1 + offset]
        ]
        face_nodePtr = np.array(face_nodePtr, dtype=int).ravel()
        offset += num_faces + 1

        face_nodesIndices = [
            line.replace("\n", "").split(" ")
            for line in lines[offset : num_face_nodes + offset]
        ]
        face_nodesIndices = np.array(face_nodesIndices, dtype=int).ravel()
        data = np.ones(face_nodesIndices.size, dtype=bool)
        face_nodes = sps.csc_matrix((data, face_nodesIndices, face_nodePtr))
        offset += num_face_nodes

        # neighbourship
        face_cells = [
            line.replace("\n", "").split(" ")
            for line in lines[offset : offset + num_faces]
        ]
        face_cells = np.array(face_cells, dtype=int).T
        offset += num_faces

        # Face areas
        face_areas = [
            line.replace("\n", "").split(" ")
            for line in lines[offset : num_faces + offset]
        ]
        face_areas = np.array(face_areas, dtype=float).ravel()
        offset += num_faces

        # Face centers
        face_centers = [
            line.replace("\n", "").split(" ")[1:]
            for line in lines[offset : offset + num_faces]
        ]
        face_centers = np.array(face_centers, dtype=float).T
        offset += num_faces

        # Face normals
        face_normals = [
            line.replace("\n", "").split(" ")[1:]
            for line in lines[offset : offset + num_faces]
        ]
        face_normals = np.array(face_normals, dtype=float).T

        offset += num_faces

        # Cell faces
        cell_facePtr = [
            line.replace("\n", "").split(" ")
            for line in lines[offset : num_cells + 1 + offset]
        ]
        cell_facePtr = np.array(cell_facePtr, dtype=int).ravel()
        offset += num_cells + 1

        cell_faces_indices = [
            line.replace("\n", "").split(" ")[1:]
            for line in lines[offset : num_cell_faces + offset]
        ]
        cell_faces_indices = np.array(cell_faces_indices, dtype=int)
        if cell_faces_indices.shape[1] == 2:
            cell_facetags = cell_faces_indices[:, 1].ravel()
            cell_faces_indices = cell_faces_indices[:, 0].ravel()
        elif cell_faces_indices.shape[1] == 1:
            cell_faces = cell_faces_indices.ravel()
        else:
            raise RuntimeError("Could not read cell_face map. Wrong dimensions")
        faces_per_cell = np.diff(cell_facePtr)
        ci = np.repeat(np.arange(num_cells), faces_per_cell)
        sgn = np.ones(num_cell_faces, dtype=int)
        sgn[face_cells[1, cell_faces_indices] == ci] = -1.0
        cell_faces = sps.csc_matrix((sgn, cell_faces_indices, cell_facePtr))

        offset += num_cell_faces

        # index map
        if has_indexmap:
            index_map = [
                line.replace("\n", "").split(" ")
                for line in lines[offset : offset + num_cells]
            ]
            index_map = np.array(index_map, dtype=int).ravel()
            offset += num_cells

        # Cell volumes
        cell_volumes = [
            line.replace("\n", "").split(" ")
            for line in lines[offset : num_cells + offset]
        ]
        cell_volumes = np.array(cell_volumes, dtype=float).ravel()
        offset += num_cells

        # Cell centers
        cell_centers = [
            line.replace("\n", "").split(" ")[1:]
            for line in lines[offset : offset + num_cells]
        ]
        cell_centers = np.array(cell_centers, dtype=float).T
        offset += num_cells

    g = pp.Grid(dim, nodes, face_nodes, cell_faces, "Unstructured grid")
    if has_indexmap:
        g.cell_index_map = index_map
    g.cell_facetag = cell_facetags
    g.face_cells = face_cells
    g.face_areas = face_areas
    g.face_normals = face_normals
    g.face_centers = face_centers
    g.cell_volumes = cell_volumes
    g.cell_centers = cell_centers
    g.cart_dims = cartDims

    return g
