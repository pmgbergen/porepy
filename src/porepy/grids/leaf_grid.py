import numpy as np
import scipy.sparse as sps
import copy
import porepy as pp


class CartLeafGrid(pp.CartGrid):
    def __init__(self, nx, physdims=None, levels=1):
        """
        Constructor for CartesianLeafGrid

        Parameters
        ----------
        nx (np.ndarray): Number of cells in each direction. Should be 1D or 2D
        physdims (np.ndarray): Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.
        levels (int): Number of grid levels. Defaults to 1
        """
        self.num_levels = levels
        self.level_grids = []
        self.mesh_sizes = []
        self.physdims = physdims
        for level in range(levels):
            self.mesh_sizes.append(np.asarray(nx) * 2 ** level)
            self.level_grids.append(pp.CartGrid(self.mesh_sizes[level], physdims))
            self.level_grids[-1].compute_geometry()
        self.name = ["CartLeafGrid"]
        self.dim = self.level_grids[0].dim

        # Number of subcell a cell is split into:
        if self.dim==2:
            self.num_subcells = 4 # for cartesian
        elif self.dim==1:
            self.num_subcells = 2
        # default to first level:
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

        self.cell_level_to_leaf = [None] * self.num_levels
        self.face_level_to_leaf = [None] * self.num_levels
        self.node_level_to_leaf = [None] * self.num_levels

        self.cell_projections_level = [None] * (self.num_levels - 1)
        self.face_projections_level = [None] * (self.num_levels - 1)
        self.node_projections_level = [None] * (self.num_levels - 1)

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
        self._init_projection()

    def copy(self):
        h = CartLeafGrid(self.mesh_sizes[0], self.physdims, self.num_levels)
        h.nodes = self.nodes.copy()
        h.num_faces = self.num_faces
        h.num_nodes = self.num_nodes
        h.num_cells = self.num_cells
        h.face_nodes = self.face_nodes.copy()
        h.cell_faces = self.cell_faces.copy()
        h.mesh_sizes = copy.deepcopy(self.mesh_sizes.copy())
        h.physdims = copy.deepcopy(self.physdims)
        h.num_levels = self.num_levels

        if hasattr(self, "cell_volumes"):
            h.cell_volumes = self.cell_volumes.copy()
        if hasattr(self, "cell_centers"):
            h.cell_centers = self.cell_centers.copy()
        if hasattr(self, "face_centers"):
            h.face_centers = self.face_centers.copy()
        if hasattr(self, "face_normals"):
            h.face_normals = self.face_normals.copy()
        if hasattr(self, "face_areas"):
            h.face_areas = self.face_areas.copy()
        if hasattr(self, "tags"):
            h.tags = self.tags.copy()
        if hasattr(self, "per_map"):
            h.per_map = self.per_map.copy()

        for level in range(self.num_levels):
            h.level_grids[level] = self.level_grids[level].copy()
        h.cell_projections = copy.deepcopy(self.cell_projections)
        h.face_projections = copy.deepcopy(self.face_projections)
        h.node_projections = copy.deepcopy(self.node_projections)

        h.cell_level_to_leaf = copy.deepcopy(self.cell_level_to_leaf)
        h.face_level_to_leaf = copy.deepcopy(self.face_level_to_leaf)
        h.node_level_to_leaf = copy.deepcopy(self.node_level_to_leaf)

        h.cell_projections_level = copy.deepcopy(self.cell_projections_level)
        h.face_projections_level = copy.deepcopy(self.face_projections_level)
        h.node_projections_level = copy.deepcopy(self.node_projections_level)

        h.block_cell_faces = self.block_cell_faces.copy()
        h.block_face_nodes = self.block_face_nodes.copy()
        h.block_nodes = copy.deepcopy(self.block_nodes)
        h.block_face_centers = copy.deepcopy(self.block_face_centers)
        h.block_face_areas = copy.deepcopy(self.block_face_areas)
        h.block_normals = copy.deepcopy(self.block_face_normals)
        h.block_cell_centers = copy.deepcopy(self.block_cell_centers)
        h.block_cell_volumes = copy.deepcopy(self.block_cell_volumes)

        h.cell_level = self.cell_level

        h.active_cells = self.active_cells
        h.active_faces = self.active_faces
        h.active_nodes = self.active_nodes
        return h

    def cell_proj_level(self, level0, level1):
        if level1 - level0 != 1:
            raise ValueError(
                "Can only calculate projection between grids 1 level apart"
            )

        if self.cell_projections_level[level0] is None:
            if self.dim==1:
                self.cell_proj_level_1d_(level0, level1)
            elif self.dim==2:
                self.cell_proj_level_2d_(level0, level1)
            else:
                raise NotImplementedError("Cell projections only implemented for 1d or 2d")

        return self.cell_projections_level[level0]


    def cell_proj_level_1d_(self, level0, level1):
        g0 = self.level_grids[level0]
        g1 = self.level_grids[level1]

        nx0 = self.mesh_sizes[level0]
        nx1 = self.mesh_sizes[level1]

        cell_indices = np.tile(np.arange(nx0), (2, 1)).ravel("F")

        cell_ptr = np.arange(g1.num_cells + 1)
        data = np.ones(cell_indices.size, dtype=int)
        self.cell_projections_level[level0] = sps.csc_matrix(
            (data, cell_indices, cell_ptr)
        )


    def cell_proj_level_2d_(self, level0, level1):
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
        self.cell_projections_level[level0] = sps.csc_matrix(
            (data, cell_indices, cell_ptr)
        )

    def face_proj_level(self, level0, level1):
        if level1 - level0 != 1:
            raise ValueError(
                "Can only calculate projection between grids 1 level apart"
            )
        if self.face_projections_level[level0] is None:
            if self.dim==1:
                self.face_projections_level[level0] = self.face_proj_level_1d_(level0, level1)
            elif self.dim==2:
                self.face_projections_level[level0] = self.face_proj_level_2d_(level0, level1)
            else:
                raise NotImplementedError("Face projections only implemented for 1d or 2d")

        return self.face_projections_level[level0]


    def face_proj_level_1d_(self, level0, level1):
        g0 = self.level_grids[level0]
        g1 = self.level_grids[level1]

        nx0 = self.mesh_sizes[level0] + 1
        nx1 = self.mesh_sizes[level1] + 1

        faces_x = np.ones(nx1, dtype=bool)
        faces_x[1::2] = False

        do_match_padded = np.r_[0, faces_x]
        indPtr = np.cumsum(do_match_padded)
        indices = np.arange(nx0)
        data = np.ones(indices.size, dtype=bool)
        return sps.csc_matrix((data, indices, indPtr))

    def face_proj_level_2d_(self, level0, level1):
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
        indices_y = np.repeat(
            np.arange(start_y, start_y + nx0[1] * (nx0[0] - 1)), 2
        )
        indices = np.r_[indices_x, indices_y]
        data = np.ones(indices.size, dtype=bool)
        return sps.csc_matrix((data, indices, indPtr))


    def node_proj_level(self, level0, level1):
        if level1 - level0 != 1:
            raise ValueError(
                "Can only calculate projection between grids 1 level apart"
            )
        if self.node_projections_level[level0] is None:
            if self.dim==1:
                self.node_projections_level[level0] = self.node_proj_level_1d_(level0, level1)
            elif self.dim==2:
                self.node_projections_level[level0] = self.node_proj_level_2d_(level0, level1)
            else:
                raise NotImplementedError("Node projections only implemented for 1d or 2d")

        return self.node_projections_level[level0]

    def node_proj_level_1d_(self, level0, level1):
        return self.face_proj_level_1d_(level0, level1)

    def node_proj_level_2d_(self, level0, level1):
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

        if cell_global.sum() == 0:
            return (
                sps.diags(
                    np.ones(self.num_cells, dtype=bool), format="csc", dtype=bool
                ),
                sps.diags(
                    np.ones(self.num_faces, dtype=bool), format="csc", dtype=bool
                )
            )

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
        old_face_proj = [None] * self.num_levels
        for level in range(self.num_levels):
            old_cell_proj[level] = self.project_level_to_leaf(level, "cell").T.copy()
            old_face_proj[level] = self.project_level_to_leaf(level, "face").T.copy()

        for level in range(self.num_levels):
            if (
                level in ref_levels
                or level - 1 in ref_levels
                or level - 2 in ref_levels
            ):
                self.update_leaf_grid(level)

        for level in range(self.num_levels):
            for entity in ["cell", "face", "node"]:
                self.update_level_to_leaf(level, entity)

        # The properites not needed in the grid refinement are only updated
        # when all cells are refined:
        self.update_grid_prop()

        # Calculate the projections from old leaf cells to new leaf cells:
        old_cell_to_new = self._calculate_projection_old_leaf_2_new_leaf(old_cell_proj, "cell")
        old_face_to_new = self._calculate_projection_old_leaf_2_new_leaf(old_face_proj, "face")

        # Propogate old face properties to new faces
        self.update_face_prop(old_face_to_new)

        return old_cell_to_new, old_face_to_new

    def coarsen_cells(self, cells):
        if self.cell_projections is None:
            self._init_projection()

        if not isinstance(np.asarray(cells).dtype, bool):
            cell_global = np.zeros(self.num_cells, dtype=bool)
            cell_global[cells] = True
        else:
            cell_global = cells

        cell_global = self.enforce_one_level_coarsening(cell_global)

        if cell_global.sum() == 0:
            old2new = sps.diags(
                np.ones(self.num_cells, dtype=bool), format="csc", dtype=bool
            )
            old_face_to_new = sps.diags(
                np.ones(self.num_faces, dtype=bool), format="csc", dtype=bool
            )
            return old2new, old_face_to_new

        # Find the current refinement level of the cells that should be coarsened
        current_level = self.cell_level[cell_global]
        # Refine the cells of each level seperately, from coarsest to finest
        coarsen_levels = np.unique(current_level)
        for level in coarsen_levels:
            leaf_to_level = self.project_level_to_leaf(level, "cell").T
            # Find cells that are on level and tagged for refinement:
            cell_p = self.cell_proj_level(level - 1, level)
            remove_cells_of_level = cell_p.T * cell_p * leaf_to_level * cell_global > self.num_subcells - 1

            # Here, we made the choice to only coarsen a cell if all (four) fine cells of the
            # coarse cell is tagged for refinement. An other option would be to refine
            # the if only one is tagged.

            # Update active cells of level
            self.active_cells[level] = self.active_cells[level] & ~remove_cells_of_level

            # The removed cells should be added to the coarse level:

            self.active_cells[level - 1] += (cell_p * remove_cells_of_level).astype(
                bool
            )
        # before we update the grid, store the old projections
        old_cell_proj = [None] * self.num_levels
        old_face_proj = [None] * self.num_levels
        for level in range(self.num_levels):
            old_cell_proj[level] = self.project_level_to_leaf(level, "cell").T.copy()
            old_face_proj[level] = self.project_level_to_leaf(level, "face").T.copy()

        must_update = np.unique(
            np.r_[coarsen_levels, coarsen_levels - 1, coarsen_levels + 1]
        )
        for level in range(self.num_levels):
            if level in must_update:
                self.update_leaf_grid(level)

        for level in range(self.num_levels):
            for entity in ["cell", "face", "node"]:
                self.update_level_to_leaf(level, entity)

        # The properites not needed in the grid refinement are only updated
        # when all cells are refined:
        self.update_grid_prop()

        # Calculate the projections from old leaf cells to new leaf cells:
        old_cell_to_new = self._calculate_projection_old_leaf_2_new_leaf(old_cell_proj, "cell")
        old_face_to_new = self._calculate_projection_old_leaf_2_new_leaf(old_face_proj, "face")

        # Propogate old face properties to new faces
        self.update_face_prop(old_face_to_new)

        return old_cell_to_new, old_face_to_new

    def update_leaf_grid(self, level):
        active_cells = self.active_cells[level]
        c2lc = _create_restriction_matrix(active_cells)
        # c2lc is short for cells_2_leaf_cells

        # Next up, the faces:
        # First, find all active faces of the current level
        active_faces = (np.abs(self.level_grids[level].cell_faces) * active_cells) > 0

        # For periodic faces, we add the faces if either side of the
        # periodic boundary is active.
        if hasattr(self, "per_map"):
            per_map = self.level_grids[level].per_map
            per_faces = active_faces[per_map[0]] | active_faces[per_map[1]]
            active_faces[per_map[0]] = per_faces
            active_faces[per_map[1]] = per_faces
        # For fracture faces, we add the faces if either side of the
        # fracture boundary is active.
        if hasattr(self, "frac_pairs"):
            frac_map = self.level_grids[level].frac_pairs
            frac_faces = active_faces[frac_map[0]] | active_faces[frac_map[1]]
            active_faces[frac_map[0]] = frac_faces
            active_faces[frac_map[1]] = frac_faces

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
            # For periodic faces or fracture faces, we add the faces if either
            # side of the boundary is active.
            if hasattr(self, "per_map"):
                per_map_fine = self.level_grids[level + 1].per_map
                per_faces_fine = (
                    faces_fine[per_map_fine[0]] | faces_fine[per_map_fine[1]]
                )
                faces_fine[per_map_fine[0]] = per_faces_fine
                faces_fine[per_map_fine[1]] = per_faces_fine
            if hasattr(self, "frac_pairs"):
                frac_map_fine = self.level_grids[level + 1].frac_pairs
                frac_faces_fine = (
                    faces_fine[frac_map_fine[0]] | faces_fine[frac_map_fine[1]]
                )
                faces_fine[frac_map_fine[0]] = frac_faces_fine
                faces_fine[frac_map_fine[1]] = frac_faces_fine
            active_faces = active_faces & (proj_f * ~faces_fine > 0)

        f2lf = _create_restriction_matrix(active_faces)
        # f2lf is short for faces_2_leaf_faces

        # Map the faces of level - 1 to the active faces of this level
        if level > 0:
            proj_f_coarse = self.face_proj_level(level - 1, level).T
            mask = sps.diags(active_faces, dtype=bool)
            cf2lf = f2lf * (mask * proj_f_coarse)  # coarse_face_2_leaf_face

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
            cn2ln = n2ln * (mask * proj_n_coarse.T)  # coarse_nodes_2_leaf_nodes

        # Update leaf grid for level
        # cell_faces and face_nodes
        cell_faces = f2lf * (self.level_grids[level].cell_faces * c2lc.T)
        face_nodes = n2ln * (self.level_grids[level].face_nodes * f2lf.T)

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
        self.active_faces[level] = active_faces
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

        if hasattr(self, "per_map"):
            self.per_map = np.ones((2, 0), dtype=int)
            for level in range(self.num_levels):
                active_faces = self.active_faces[level]
                left_idx = self.level_grids[level].per_map[0]
                right_idx = self.level_grids[level].per_map[1]
                # The left faces and right faces must be given in strictly
                # increasing order. The reason for this is that when we map them
                # to the leaf grid using boolean arrays, we loose the ordering.
                # For Cartesian grids this should not be any restriction, for simplices
                # you have to do something smarter
                if not (left_idx == np.sort(left_idx)).all():
                    raise ValueError("Can only map periodic boundaries that are sorted")
                if not (right_idx == np.sort(right_idx)).all():
                    raise ValueError("Can only map periodic boundaries that are sorted")

                left_level = np.zeros(self.level_grids[level].num_faces, dtype=bool)
                right_level = np.zeros(self.level_grids[level].num_faces, dtype=bool)

                left_level[left_idx] = True
                right_level[right_idx] = True

                level_to_leaf = self.project_level_to_leaf(level, "face")

                left_leaf = np.argwhere(level_to_leaf * left_level).ravel()
                right_leaf = np.argwhere(level_to_leaf * right_level).ravel()

                per_map_level = np.vstack((left_leaf, right_leaf))
                self.per_map = np.c_[self.per_map, per_map_level]

    def update_face_prop(self, old_to_new_face):
        for key in self.tags.keys():
            if self.tags[key].size == old_to_new_face.shape[1]:
                self.tags[key] = (old_to_new_face * self.tags[key]) > 0

        if hasattr(self, "frac_pairs"):
            for i in range(len(self.frac_pairs)):
                side = np.zeros(old_to_new_face.shape[1], dtype=int)
                side[self.frac_pairs[i]] = True
                self.frac_pairs[i] = np.argwhere(old_to_new_face * side).ravel()
                

    def project_level_to_leaf(self, level, element_type):
        if element_type == "cell":
            return self.cell_level_to_leaf[level]
        elif element_type == "face":
            return self.face_level_to_leaf[level]
        elif element_type == "node":
            return self.node_level_to_leaf[level]
        else:
            raise ValueError(
                "Unknwon element_type: {}. Possible choises are: cell, face, node".format(
                    element_type
                )
            )

    def update_level_to_leaf(self, level, element_type):
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

        if element_type == "cell":
            self.cell_level_to_leaf[level] = sps.vstack(
                block_mat, format="csc", dtype=bool
            )
        elif element_type == "face":
            self.face_level_to_leaf[level] = sps.vstack(
                block_mat, format="csc", dtype=bool
            )
        elif element_type == "node":
            self.node_level_to_leaf[level] = sps.vstack(
                block_mat, format="csc", dtype=bool
            )
        else:
            raise ValueError(
                "Unknwon element_type: {}. Possible choises are: cell, face, node".format(
                    element_type
                )
            )

    def enforce_one_level_refinement(self, cell_global):
        """
        We make sure that two neighbour cells are at most 1 level appart after refinement
        """
        if cell_global.sum() == 0:
            return cell_global

        # no need to check levels less than maximm to refine
        max_level = np.max(self.cell_level[cell_global])
        for level in range(max_level, 0, -1):
            cells_of_level = self.cell_level == level
            # Find cells that are on level and tagged for refinement:
            level_cells_to_ref = np.zeros(cell_global.size, dtype=bool)
            level_cells_to_ref[cells_of_level] = cell_global[cells_of_level]

            # Project to faces and back to cells to get neighbour cells
            neighbour_mapping = np.abs(self.cell_faces.T) * np.abs(self.cell_faces)
            # Refine over periodic boundary:
            if hasattr(self, "per_map"):
                is_left = np.zeros(self.num_faces, dtype=bool)
                is_right = np.zeros(self.num_faces, dtype=bool)
                is_left[self.per_map[0]] = True
                is_right[self.per_map[1]] = True
                col = self.per_map[0]
                row = self.per_map[1]
                data = np.ones(self.per_map.shape[1], dtype=bool)
                shape = (self.num_faces, self.num_faces)
                left_to_right = sps.coo_matrix((data, (row, col)), shape=shape)
                per_face_map = left_to_right + left_to_right.T
                per_cell_map = (
                    np.abs(self.cell_faces.T) * per_face_map * np.abs(self.cell_faces)
                )
                neighbour_mapping += per_cell_map

            # Refine over fractures
            if hasattr(self, "frac_pairs"):
                is_left = np.zeros(self.num_faces, dtype=bool)
                is_right = np.zeros(self.num_faces, dtype=bool)
                is_left[self.frac_pairs[0]] = True
                is_right[self.frac_pairs[1]] = True
                col = self.frac_pairs[0]
                row = self.frac_pairs[1]
                data = np.ones(self.frac_pairs[0].size, dtype=bool)
                shape = (self.num_faces, self.num_faces)
                left_to_right = sps.coo_matrix((data, (row, col)), shape=shape)
                per_face_map = left_to_right + left_to_right.T
                per_cell_map = (
                    np.abs(self.cell_faces.T) * per_face_map * np.abs(self.cell_faces)
                )
                neighbour_mapping += per_cell_map

            neighbours = (neighbour_mapping * level_cells_to_ref) > 0

            refine_neighbours = np.zeros(self.num_cells, dtype=bool)
            refine_neighbours[neighbours] = self.cell_level[neighbours] < level

            cell_global = cell_global | refine_neighbours

        return cell_global

    def enforce_one_level_coarsening(self, cell_global):
        """
        We make sure that two neighbour cells are at most 1 level appart after coarsening
        """
        if cell_global.sum() == 0:
            return cell_global
        # no need to check levels less than maximum to refine
        max_level = np.max(self.cell_level[cell_global])
        cell_level = self.cell_level.copy()
        for level in range(max_level, 0, -1):
            cells_of_level = self.cell_level == level

            leaf_to_level = self.project_level_to_leaf(level, "cell").T
            # Find cells that are on level and tagged for refinement:
            cell_p = self.cell_proj_level(level - 1, level)
            remove_cells_of_level = leaf_to_level.T * (
                cell_p.T * cell_p * leaf_to_level * cell_global > 3
            )

            # Project to faces and back to cells to get neighbour cells
            neighbour_mapping = np.abs(self.cell_faces.T) * np.abs(self.cell_faces)
            # Add neighbourship over periodic boundary:
            if hasattr(self, "per_map"):
                is_left = np.zeros(self.num_faces, dtype=bool)
                is_right = np.zeros(self.num_faces, dtype=bool)
                is_left[self.per_map[0]] = True
                is_right[self.per_map[1]] = True
                col = self.per_map[0]
                row = self.per_map[1]
                data = np.ones(self.per_map.shape[1], dtype=bool)
                shape = (self.num_faces, self.num_faces)
                left_to_right = sps.coo_matrix((data, (row, col)), shape=shape)
                per_face_map = left_to_right + left_to_right.T
                per_cell_map = (
                    np.abs(self.cell_faces.T) * per_face_map * np.abs(self.cell_faces)
                )
                neighbour_mapping += per_cell_map

                # Add neighbourship over fracture boundary:
            if hasattr(self, "frac_pairs"):
                is_left = np.zeros(self.num_faces, dtype=bool)
                is_right = np.zeros(self.num_faces, dtype=bool)
                is_left[self.frac_pairs[0]] = True
                is_right[self.frac_pairs[1]] = True
                col = self.frac_pairs[0]
                row = self.frac_pairs[1]
                data = np.ones(self.frac_pairs[0].size, dtype=bool)
                shape = (self.num_faces, self.num_faces)
                left_to_right = sps.coo_matrix((data, (row, col)), shape=shape)
                per_face_map = left_to_right + left_to_right.T
                per_cell_map = (
                    np.abs(self.cell_faces.T) * per_face_map * np.abs(self.cell_faces)
                )
                neighbour_mapping += per_cell_map

            neighbours = (neighbour_mapping * remove_cells_of_level) > 0

            to_fine_neighbours = cell_level > level
            to_fine_neighbours = (neighbour_mapping * to_fine_neighbours) > 0
            to_fine_neighbours[~cells_of_level] = False
            cell_global = cell_global & ~to_fine_neighbours
        #            cell_level[remove_cells_of_level & ~to_fine_neighbours] = 1
        #            import pdb; pdb.set_trace()

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
        for level in range(self.num_levels):
            self.update_level_to_leaf(level, "cell")

        proj_f = [None] * self.num_levels
        proj_f[0] = sps.diags(
            np.ones(self.level_grids[0].num_faces, dtype=bool), dtype=bool
        )
        for level, g in enumerate(self.level_grids[1:]):
            proj_f[level + 1] = sps.csc_matrix((0, g.num_faces))
        self.face_projections = proj_f
        for level in range(self.num_levels):
            self.update_level_to_leaf(level, "face")

        proj_n = [None] * self.num_levels
        proj_n[0] = sps.diags(
            np.ones(self.level_grids[0].num_nodes, dtype=bool), dtype=bool
        )
        for level, g in enumerate(self.level_grids[1:]):
            proj_n[level + 1] = sps.csc_matrix((0, g.num_nodes))
        self.node_projections = proj_n
        for level in range(self.num_levels):
            self.update_level_to_leaf(level, "node")

        proj_o2n = [None] * self.num_levels
        proj_o2n[0] = sps.diags(np.ones(self.num_cells, dtype=bool), dtype=bool)
        # for level, g in enumerate(self.level_grids[1:]):
        #     proj_o2n[level + 1] = sps.csc_matrix((0, self.num_cells))
        self.old_to_new = proj_o2n

        for level in range(self.num_levels - 1):
            self.cell_proj_level(level, level + 1)
            self.face_proj_level(level, level + 1)
            self.node_proj_level(level, level + 1)


    def _calculate_projection_old_leaf_2_new_leaf(self, old_proj, element_type="cell"):
        if element_type=="cell":
            number_of_elements = self.num_cells
        elif element_type=="face":
            number_of_elements = self.num_faces
        else:
            raise ValueError("Unknown type: " + str(element_type))
        old_to_new = sps.csc_matrix(
            (number_of_elements, old_proj[0].shape[1]), dtype=bool
        )
        for level in range(self.num_levels):
            # Add old leaf elements of level that are also in the new leaf elements
            old_to_new += (
                self.project_level_to_leaf(level, element_type) * old_proj[level]
            )
            if level < self.num_levels - 1:
                # Add old leaf elements that has been refined:
                if element_type=="cell":
                    coarse_level_to_fine = self.cell_proj_level(level, level + 1).T
                elif element_type=="face":
                    coarse_level_to_fine = self.face_proj_level(level, level + 1).T
                else:
                    raise ValueError("Unknown type: " + str(element_type))

                old_leaf_to_fine = coarse_level_to_fine * old_proj[level]
                old_to_new += (
                    self.project_level_to_leaf(level + 1, element_type) * old_leaf_to_fine
                )
            if level > 0:
                # Add old leaf elements that has been coarsened:
                if element_type=="cell":
                    fine_level_to_coarse = self.cell_proj_level(level - 1, level)
                elif element_type=="face":
                    fine_level_to_coarse = self.face_proj_level(level - 1, level)
                else:
                    raise ValueError("Unknown type: " + str(element_type))

                old_leaf_to_coarse = fine_level_to_coarse * old_proj[level]
                old_to_leaf = (
                    self.project_level_to_leaf(level - 1, element_type) * old_leaf_to_coarse
                )
                # When coarsening an element we take the average
                weight = np.bincount(old_to_leaf.indices)
                _, IA = np.unique(old_to_leaf.indices, return_inverse=True)
                weight = np.bincount(IA)
                old_to_leaf.data = old_to_leaf.data / weight[IA]
                old_to_new += old_to_leaf

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
    lg = pp.CartLeafGrid([10, 5], [1, 1], 3)
    print("time to generate leaf grid: {} s".format(time.time() - tic))
    for level in range(lg.num_levels):
        left = np.argwhere(lg.level_grids[level].face_centers[1] < 1e-5).ravel()
        right = np.argwhere(lg.level_grids[level].face_centers[1] > 1 - 1e-5).ravel()
        lg.level_grids[level].per_map = np.vstack((left, right))

    tic = time.time()
    tag = (0.3 < lg.cell_centers[0]) & (lg.cell_centers[0] < 0.7)
    lg.refine_cells(tag)

    lg_ref = pp.CartLeafGrid([2, 2], [1, 1], 3)
    lg_ref.refine_cells(0)
    lg_ref.refine_cells([1, 3])
    lg.compute_geometry()
    #    lg.refine_cells(3)
    #    lg.refine_cells()
    #    lg.compute_geometry()
    print("time to refine leaf grid: {} s".format(time.time() - tic))
    if True:
        pp.plot_grid(lg, if_plot=False, alpha=0)

        plt.plot(lg.cell_centers[0], lg.cell_centers[1], ".")
        plt.plot(lg.nodes[0], lg.nodes[1], "o")
        plt.plot(lg.face_centers[0], lg.face_centers[1], "x")
        plt.show()

    print(lg.per_map)
    pp.plot_grid(lg, info="f", alpha=0, if_plot=False)
    ax = plt.gca()
    plt.axis("off")
    plt.show()
    left = lg.per_map[0]
    right = lg.per_map[1]
    lg.face_centers[:, left]
    lg.face_centers[:, right]
