import numpy as np
import scipy.sparse as sps
import porepy as pp


class CartLeafGrid(pp.CartGrid):
    def __init__(self, nx, physdims=None, levels=1):
        """
        Constructor for CartesianLeafGrid

        Parameters
        ----------
        nx (np.ndarray): Number of cells in each direction. Should be 2D or 3D
        physdims (np.ndarray): Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.
        levels (int): Number of grid levels. Defaults to 1
        """
        self.num_levels = levels
        self.level_grids = []
        self.mesh_sizes = []
        for level in range(levels):
            self.mesh_sizes.append(np.asarray(nx) * 2 ** level)
            self.level_grids.append(pp.CartGrid(self.mesh_sizes[level], physdims))
            self.level_grids[-1].compute_geometry()
        self.name = ["CartLeafGrid"]

        # default to first level:
        self.dim = self.level_grids[0].dim
        self.nodes = self.level_grids[0].nodes.copy()
        self.cell_faces = self.level_grids[0].cell_faces.copy()
        self.face_nodes = self.level_grids[0].face_nodes.copy()
        self.num_nodes = self.level_grids[0].num_nodes
        self.num_faces = self.level_grids[0].num_faces
        self.num_cells = self.level_grids[0].num_cells
        self.tags = self.level_grids[0].tags.copy()

        self.nodes = self.level_grids[0].nodes.copy()
        self.face_normals = self.level_grids[0].face_normals.copy()
        self.face_areas = self.level_grids[0].face_areas.copy()
        self.face_centers = self.level_grids[0].face_centers.copy()
        self.cell_volumes = self.level_grids[0].cell_volumes.copy()
        self.cell_centers = self.level_grids[0].cell_centers.copy()

        self.cell_projections = None
        self.face_projections = None
        self.node_projections = None

        self.block_cell_faces = np.empty((levels, levels), dtype=object)
        self.block_face_nodes = np.empty((levels, levels), dtype=object)
        self.block_nodes = [np.ones((3, 0))] * levels
        self.block_face_centers = [np.ones((3, 0))] * levels
        self.block_face_areas = [np.ones(0)] * levels
        self.block_face_normals = [np.ones((3, 0))] * levels
        self.block_cell_centers = [np.ones((3, 0))] * levels
        self.block_cell_volumes = [np.ones(0)] * levels

        self.block_cell_faces[0, 0] = self.cell_faces
        self.block_face_nodes[0, 0] = self.face_nodes
        self.block_nodes[0] = self.nodes
        self.block_face_areas[0] = self.face_areas
        self.block_face_centers[0] = self.face_centers
        self.block_face_normals[0] = self.face_normals
        self.block_cell_volumes[0] = self.cell_volumes
        self.block_cell_centers[0] = self.cell_centers

        self.cell_level = np.zeros(self.num_cells, dtype=int)

        self.active_cells = None
        self.active_faces = None
        self.active_nodes = None
        self._init_active_cells()

    def cell_proj_level(self, level0, level1):
        if level1 - level0 != 1:
            raise ValueError(
                "Can only calculate projection between grids 1 level apart"
            )

        g0 = self.level_grids[level0]
        g1 = self.level_grids[level1]

        nx0 = self.mesh_sizes[level0]
        nx1 = self.mesh_sizes[level1]

        offset = np.atleast_2d(np.cumsum([nx0[0]] * nx0[1])).T
        offset -= offset[0]
        cell_indices = np.tile(
            np.tile(np.arange(nx0[0]), (2, 2)).ravel("F"), (nx0[1], 1)
        )
        cell_indices += offset
        cell_indices = cell_indices.ravel("C")

        cell_ptr = np.arange(g1.num_cells + 1)
        data = np.ones(cell_indices.size, dtype=int)
        return sps.csc_matrix((data, cell_indices, cell_ptr))

    def face_proj_level(self, level0, level1):
        if level1 - level0 != 1:
            raise ValueError(
                "Can only calculate projection between grids 1 level apart"
            )

        g0 = self.level_grids[level0]
        g1 = self.level_grids[level1]

        nx0 = self.mesh_sizes[level0] + 1
        nx1 = self.mesh_sizes[level1] + 1

        faces_x = np.ones(nx1[0], dtype=bool)
        faces_x[1::2] = False
        faces_y = np.ones(nx1[1] - 1, dtype=bool)
        faces_XX, faces_YY = np.meshgrid(faces_x, faces_y)
        do_match_x = faces_XX.flatten() * faces_YY.flatten()

        faces_x = np.ones(nx1[0] - 1, dtype=bool)
        faces_y = np.ones(nx1[1], dtype=bool)
        faces_y[1::2] = False
        faces_XX, faces_YY = np.meshgrid(faces_x, faces_y)
        do_match_y = faces_XX.flatten() * faces_YY.flatten()

        do_match = np.r_[do_match_x, do_match_y]

        do_match_padded = np.r_[0, do_match]
        indPtr = np.cumsum(do_match_padded)
        indices_x_row = np.arange(nx0[0] * (nx0[1] - 1)).reshape((-1, nx0[0]))
        indices_x = np.tile(indices_x_row, (1, 2)).ravel()
        start_y = nx0[0] * (nx0[1] - 1)
        indices_y = np.repeat(np.arange(start_y, start_y + nx0[1] * (nx0[0] - 1)), 2)
        indices = np.r_[indices_x, indices_y]
        data = np.ones(indices.size, dtype=bool)

        return sps.csc_matrix((data, indices, indPtr))

    def node_proj_level(self, level0, level1):
        if level1 - level0 != 1:
            raise ValueError(
                "Can only calculate projection between grids 1 level apart"
            )

        g0 = self.level_grids[level0]
        g1 = self.level_grids[level1]

        nx0 = self.mesh_sizes[level0] + 1
        nx1 = self.mesh_sizes[level1] + 1

        nodes_x = np.ones(nx1[0], dtype=bool)
        nodes_x[1::2] = False
        nodes_y = np.ones(nx1[1], dtype=bool)
        nodes_y[1::2] = False

        nodes_XX, nodes_YY = np.meshgrid(nodes_x, nodes_y)

        do_match = nodes_XX.flatten() * nodes_YY.flatten()

        do_match_padded = np.r_[0, do_match]
        indPtr = np.cumsum(do_match_padded)
        indices = np.arange(do_match.sum())
        data = np.ones(indices.size, dtype=bool)
        return sps.csc_matrix((data, indices, indPtr))

    def refine_cells(self, cells):
        if self.cell_projections is None:
            self._init_projection()

        if not isinstance(np.asarray(cells).dtype, bool):
            cell_global = np.zeros(self.num_cells, dtype=bool)
            cell_global[cells] = True
        else:
            cell_global = cells

        # Make sure neighbour cells are at most seperated by one level:
        cell_global = self.enforce_one_level_refinement(cell_global)

        # Find the current refinement level of the cells that should be refined
        current_level = self.cell_level[cell_global]
        # The matrix old_to_new gives the mapping from old leaf cells to new leaf cells
        old_to_new = sps.eye(self.num_cells, dtype=bool)
        # Refine the cells of each level seperately, from coarsest to finest
        ref_levels = np.unique(current_level)
        for level in ref_levels:
            leaf_to_level = self.project_level_to_leaf(level, "cell").T
            # Find cells that are on level and tagged for refinement:
            remove_cells_of_level = leaf_to_level * cell_global
            # In addition to the tagged cells, we also refine all cells
            # that have previously been refined. That is, they have a level, less
            # than the coarse cell
            self.active_cells[level] = self.active_cells[level] & ~remove_cells_of_level

            # The removed cells should be added to the next level:
            cell_p = self.cell_proj_level(level, level + 1)
            self.active_cells[level + 1] += (cell_p.T * remove_cells_of_level).astype(
                bool
            )

        # before we update the grid, store the old projections
        old_cell_proj = [None] * self.num_levels
        for level in range(self.num_levels):
            old_cell_proj[level] = self.project_level_to_leaf(level, 'cell').T.copy()

        for level in range(self.num_levels):
            self.update_leaf_grid(level)

        # The properites not needed in the grid refinement are only updated
        # when all cells are refined:
        self.update_grid_prop()

        # Calculate the projections from old leaf cells to new leaf cells:
        old_to_new = self._calculate_projection_old_leaf_2_new_leaf(old_cell_proj)

        return old_to_new

    def coarsen_cells(self, cells):
        if self.cell_projections is None:
            self._init_projection()

        if not isinstance(np.asarray(cells).dtype, bool):
            cell_global = np.zeros(self.num_cells, dtype=bool)
            cell_global[cells] = True
        else:
            cell_global = cells

        cell_global = self.enforce_one_level_coarsening(cell_global)
        # Find the current refinement level of the cells that should be coarsened
        current_level = self.cell_level[cell_global]
        # Refine the cells of each level seperately, from coarsest to finest
        coarsen_levels = np.unique(current_level)
        for level in coarsen_levels:
            leaf_to_level = self.project_level_to_leaf(level, "cell").T
            # Find cells that are on level and tagged for refinement:
            cell_p = self.cell_proj_level(level - 1, level)
            remove_cells_of_level = cell_p.T * cell_p * leaf_to_level * cell_global > 3
            # Here, we made the choice to only coarsen a cell if all (four) fine cells of the
            # coarse cell is tagged for refinement. An other option would be to refine
            # the if only one is tagged.

            # Update active cells of level
            self.active_cells[level] = self.active_cells[level] & ~remove_cells_of_level

            # The removed cells should be added to the coarse level:

            self.active_cells[level - 1] += (cell_p * remove_cells_of_level).astype(
                bool
            )

        for level in range(self.num_levels):
            old_to_new = self.update_leaf_grid(level)

        # The properites not needed in the grid refinement are only updated
        # when all cells are refined:
        self.update_grid_prop()

    def update_leaf_grid(self, level):
        active_cells = self.active_cells[level]
        c2lc = _create_restriction_matrix(active_cells)
        # c2lc is short for cells_2_leaf_cells

        # Next up, the faces:
        # First, find all active faces of the current level
        active_faces = (np.abs(self.level_grids[level].cell_faces) * active_cells) > 0

        if level < self.num_levels - 1:
            # Not all of these should be included. If a neighbour cell is refined,
            # we should use the refined faces instead:
            proj_f = self.face_proj_level(
                level, level + 1
            )  # projection from fine faces to coarse
            cells_fine = self.active_cells[level + 1]
            faces_fine = (
                np.abs(self.level_grids[level + 1].cell_faces) * cells_fine
            ) > 0
            active_faces = active_faces & (proj_f * ~faces_fine > 0)

        f2lf = _create_restriction_matrix(active_faces)
        # f2lf is short for faces_2_leaf_faces

        # Map the faces of level - 1 to the active faces of this level
        if level > 0:
            proj_f_coarse = self.face_proj_level(level - 1, level)
            mask = sps.diags(active_faces, dtype=bool)
            cf2lf = f2lf * mask * proj_f_coarse.T  # coarse_face_2_leaf_face

        # As for the faces, we only pick nodes that are not in a finer level
        active_nodes = (np.abs(self.level_grids[level].face_nodes) * active_faces) > 0
        if level < self.num_levels - 1:
            proj_n = self.node_proj_level(
                level, level + 1
            )  # projection from fine nodes to coarse
            nodes_fine = (
                np.abs(self.level_grids[level + 1].face_nodes) * faces_fine
            ) > 0
            active_nodes = active_nodes & (proj_n * ~nodes_fine > 0)

        # Map from level nodes to leaf nodes
        n2ln = _create_restriction_matrix(active_nodes)
        # n2ln is short for nodes_2_leaf_nodes

        # Map the nodes of this level to the faces of level - 1.
        if level > 0:
            proj_n_coarse = self.node_proj_level(level - 1, level)
            mask = sps.diags(active_nodes, dtype=bool)
            cn2ln = n2ln * mask * proj_n_coarse.T  # coarse_nodes_2_leaf_nodes

        # Update leaf grid for level
        # cell_faces and face_nodes
        cell_faces = f2lf * self.level_grids[level].cell_faces * c2lc.T
        face_nodes = n2ln * self.level_grids[level].face_nodes * f2lf.T
        self.block_cell_faces[level, level] = cell_faces
        self.block_face_nodes[level, level] = face_nodes

        # Update mappings between levels:
        if level > 0:
            face_nodes_coarse = (
                cn2ln
                * self.level_grids[level - 1].face_nodes
                * self.face_projections[level - 1].T
            )
            cell_faces_coarse = (
                cf2lf
                * self.level_grids[level - 1].cell_faces
                * self.cell_projections[level - 1].T
            )
            self.block_cell_faces[level, level - 1] = cell_faces_coarse
            self.block_face_nodes[level, level - 1] = face_nodes_coarse

        # Update leaf grid geometry:
        self.block_nodes[level] = self.level_grids[level].nodes[:, active_nodes]
        self.block_face_centers[level] = self.level_grids[level].face_centers[
            :, active_faces
        ]
        self.block_face_normals[level] = self.level_grids[level].face_normals[
            :, active_faces
        ]
        self.block_face_areas[level] = self.level_grids[level].face_areas[active_faces]
        self.block_cell_volumes[level] = self.level_grids[level].cell_volumes[
            active_cells
        ]
        self.block_cell_centers[level] = self.level_grids[level].cell_centers[
            :, active_cells
        ]

        # Store projections:
        self.cell_projections[level] = c2lc
        self.face_projections[level] = f2lf
        self.node_projections[level] = n2ln

    def new_cell_projection(self):
        return sps.vstack(self.old_to_new)

    def update_grid_prop(self):
        self.cell_faces = sps.bmat(self.block_cell_faces, format="csc")
        self.face_nodes = sps.bmat(self.block_face_nodes, format="csc")
        self.nodes = np.hstack(self.block_nodes)

        self.face_centers = np.hstack(self.block_face_centers)
        self.face_normals = np.hstack(self.block_face_normals)
        self.face_areas = np.hstack(self.block_face_areas)
        self.cell_volumes = np.hstack(self.block_cell_volumes)
        self.cell_centers = np.hstack(self.block_cell_centers)
        self.num_cells = self.cell_volumes.size
        self.num_faces = self.face_areas.size
        self.num_nodes = self.nodes.shape[1]
        self.cell_level = np.hstack(
            [
                i * np.ones(cc.shape[1], dtype=int)
                for i, cc in enumerate(self.block_cell_centers)
                if cc is not None
            ]
        )

        self.initiate_face_tags()
        self.initiate_node_tags()
        self.update_boundary_face_tag()

    def project_level_to_leaf(self, level, element_type):
        if element_type == "cell":
            active = self.cell_projections
            num_elements = self.level_grids[level].num_cells
        elif element_type == "face":
            active = self.face_projections
            num_elements = self.level_grids[level].num_faces
        elif element_type == "node":
            active = self.node_projections
            num_elements = self.level_grids[level].num_nodes
        else:
            raise ValueError(
                "Unknwon element_type: {}. Possible choises are: cell, face, node".format(
                    element_type
                )
            )

        block_mat = np.empty(self.num_levels, dtype=object)
        for i in range(self.num_levels):
            if i == level:
                block_mat[i] = active[level]
            else:
                block_mat[i] = sps.csc_matrix((active[i].shape[0], num_elements))
        return sps.vstack(block_mat, format="csc", dtype=bool)

    def enforce_one_level_refinement(self, cell_global):
        """
        We make sure that two neighbour cells are at most 1 level appart after refinement
        """

        # no need to check levels less than maximm to refine
        max_level = np.max(self.cell_level[cell_global])
        for level in range(max_level, 0, -1):
            cells_of_level = self.cell_level == level
            # Find cells that are on level and tagged for refinement:
            level_cells_to_ref = np.zeros(cell_global.size, dtype=bool)
            level_cells_to_ref[cells_of_level] = cell_global[cells_of_level]

            # Project to faces and back to cells to get neighbour cells
            neighbour_mapping = np.abs(self.cell_faces.T) * np.abs(self.cell_faces)
            neighbours = (neighbour_mapping * level_cells_to_ref) > 0

            refine_neighbours = np.zeros(self.num_cells, dtype=bool)
            refine_neighbours[neighbours] = self.cell_level[neighbours] < level

            cell_global = cell_global | refine_neighbours

        return cell_global

    def enforce_one_level_coarsening(self, cell_global):
        """
        We make sure that two neighbour cells are at most 1 level appart after coarsening
        """

        # no need to check levels less than maximum to refine
        max_level = np.max(self.cell_level[cell_global])
        for level in range(max_level, 0, -1):
            cells_of_level = self.cell_level == level
            # Find cells that are on level and tagged for refinement:
            level_cells_to_ref = np.zeros(cell_global.size, dtype=bool)
            level_cells_to_ref[cells_of_level] = cell_global[cells_of_level]

            # Project to faces and back to cells to get neighbour cells
            neighbour_mapping = np.abs(self.cell_faces.T) * np.abs(self.cell_faces)
            neighbours = (neighbour_mapping * level_cells_to_ref) > 0

            to_fine_neighbours = np.zeros(self.num_cells, dtype=bool)
            to_fine_neighbours[neighbours] = self.cell_level[neighbours] > level

            cell_global = cell_global & ~((neighbour_mapping * to_fine_neighbours) > 0)

        return cell_global

    def _init_active_cells(self):
        # Inital grid is the coarse grid:
        active_c = [None] * self.num_levels
        active_c[0] = np.ones(self.num_cells, dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            active_c[level + 1] = np.zeros(g.num_cells, dtype=bool)
        self.active_cells = active_c

        active_f = [None] * self.num_levels
        active_f[0] = sps.diags(
            np.ones(self.level_grids[0].num_faces, dtype=bool), dtype=bool
        )
        for level, g in enumerate(self.level_grids[1:]):
            active_f[level + 1] = sps.csc_matrix((g.num_faces, g.num_faces), dtype=bool)
        self.active_faces = active_f

        active_n = [None] * self.num_levels
        active_n[0] = sps.diags(
            np.ones(self.level_grids[0].num_nodes, dtype=bool), dtype=bool
        )
        for level, g in enumerate(self.level_grids[1:]):
            active_n[level + 1] = sps.csc_matrix((g.num_nodes, g.num_nodes), dtype=bool)
        self.active_nodes = active_n

    def _init_projection(self):
        # Inital grid is the coarse grid:
        proj_c = [None] * self.num_levels
        proj_c[0] = sps.diags(np.ones(self.num_cells, dtype=bool), dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            proj_c[level + 1] = sps.csc_matrix((0, g.num_cells))
        self.cell_projections = proj_c

        proj_f = [None] * self.num_levels
        proj_f[0] = sps.diags(
            np.ones(self.level_grids[0].num_faces, dtype=bool), dtype=bool
        )
        for level, g in enumerate(self.level_grids[1:]):
            proj_f[level + 1] = sps.csc_matrix((0, g.num_faces))
        self.face_projections = proj_f

        proj_n = [None] * self.num_levels
        proj_n[0] = sps.diags(
            np.ones(self.level_grids[0].num_nodes, dtype=bool), dtype=bool
        )
        for level, g in enumerate(self.level_grids[1:]):
            proj_n[level + 1] = sps.csc_matrix((0, g.num_nodes))
        self.node_projections = proj_n

        proj_o2n = [None] * self.num_levels
        proj_o2n[0] = sps.diags(np.ones(self.num_cells, dtype=bool), dtype=bool)
        # for level, g in enumerate(self.level_grids[1:]):
        #     proj_o2n[level + 1] = sps.csc_matrix((0, self.num_cells))
        self.old_to_new = proj_o2n

    def _calculate_projection_old_leaf_2_new_leaf(self, old_cell_proj):
        old_to_new = sps.csc_matrix((self.num_cells, old_cell_proj[0].shape[1]), dtype=bool)
        for level in range(self.num_levels):
            # Add old leaf cells of level that are also in the new leaf cells
            old_to_new += self.project_level_to_leaf(level, 'cell') * old_cell_proj[level]
            if level < self.num_levels - 1:
                # Add old leaf cells that has been refined:
                coarse_level_to_fine = self.cell_proj_level(level, level + 1).T
                old_leaf_to_fine =  coarse_level_to_fine * old_cell_proj[level]
                old_to_new += self.project_level_to_leaf(level + 1, 'cell') * old_leaf_to_fine
        return old_to_new

def _create_restriction_matrix(keep):
    size = keep.sum()
    indices = np.arange(size)
    indptr = np.zeros(keep.size + 1, dtype=int)
    indptr[1:][keep] = 1
    indptr = np.cumsum(indptr)
    data = np.ones(indices.size, dtype=bool)
    return sps.csc_matrix((data, indices, indptr), shape=(size, keep.size))


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    tic = time.time()
    lg = pp.CartLeafGrid([1, 2], [1, 1], 3)
    print("time to generate leaf grid: {} s".format(time.time() - tic))

    tic = time.time()

    old_to_new0 = lg.refine_cells(0)
    old_to_new1 = lg.refine_cells([0, 1])

    #    lg.compute_geometry()
    print("time to refine leaf grid: {} s".format(time.time() - tic))
    if False:
        plt.plot(lg.cell_centers[0], lg.cell_centers[1], ".")
        plt.plot(lg.nodes[0], lg.nodes[1], "o")
        plt.plot(lg.face_centers[0], lg.face_centers[1], "x")
        plt.show()

    pp.plot_grid(lg)
