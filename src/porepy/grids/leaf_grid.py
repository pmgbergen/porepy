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

        self.level_grids = []
        self.mesh_sizes = []
        for level in range(levels):
            self.mesh_sizes.append( np.asarray(nx) * 2 **level )
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
        if level1 -level0 != 1:
            raise ValueError("Can only calculate projection between grids 1 level apart")

        g0 = self.level_grids[level0]
        g1 = self.level_grids[level1]

        nx0 = self.mesh_sizes[level0]
        nx1 = self.mesh_sizes[level1]

        
        offset = np.atleast_2d(np.cumsum([nx0[0]] * nx0[1])).T
        offset -= offset[0]
        cell_indices = np.tile(np.tile(np.arange(nx0[0]), (2, 2)).ravel('F'), (nx0[1], 1))
        cell_indices += offset
        cell_indices = cell_indices.ravel('C')

        cell_ptr = np.arange(g1.num_cells + 1)
        data = np.ones(cell_indices.size, dtype=int)
        return sps.csc_matrix((data, cell_indices, cell_ptr))

    def face_proj_level(self, level0, level1):
        if level1 -level0 != 1:
            raise ValueError("Can only calculate projection between grids 1 level apart")

        g0 = self.level_grids[level0]
        g1 = self.level_grids[level1]

        nx0 = self.mesh_sizes[level0] + 1
        nx1 = self.mesh_sizes[level1] + 1

        faces_x = np.ones(nx1[0], dtype=bool)
        faces_x[1::2] = False
        faces_y = np.ones(nx1[1] - 1, dtype=bool)
        faces_XX, faces_YY = np.meshgrid(faces_x, faces_y)
        do_match_x = faces_XX.flatten() * faces_YY.flatten()

        faces_x = np.ones(nx1[0] -1, dtype=bool)
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
        if level1 -level0 != 1:
            raise ValueError("Can only calculate projection between grids 1 level apart")

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

        min_level = np.min(self.cell_level)
        max_level = np.max(self.cell_level)

        ref_levels = np.unique(self.cell_level[cell_global] + 1)

        if ref_levels.size > 1:
            raise NotImplementedError('refine_cells can only refine cells at the same level')

        max_to_refine = max(np.max(ref_levels), np.max(self.cell_level))
        
        level = ref_levels[0] -1
        refine_cells = cell_global | (self.cell_level > level)

        self.refine_level(level, refine_cells)
        for level in range(level + 1, np.max(self.cell_level)):
            self.refine_level(level, [])

        self.update_grid_prop()


    def refine_level(self, level, cells):

        if not isinstance(np.asarray(cells).dtype, bool):
            cells_leaf_to_ref = np.zeros(self.num_cells, dtype=bool)
            cells_leaf_to_ref[cells] = True
        else:
            cells_leaf_to_ref = cells

        cell_to_leaf_c = self.project_level_to_leaf(level, 'cell')
        cell_to_leaf_f = self.project_level_to_leaf(level + 1, 'cell')

        cells_c = cell_to_leaf_c.T * ~cells_leaf_to_ref

        proj_c = self.cell_proj_level(level, level + 1)
        proj_f = self.face_proj_level(level, level + 1)
        proj_n = self.node_proj_level(level, level + 1)

#        cells_f = proj_c.T * ~cells_c > 0
        # Find fine cells
        to_refine = proj_c.T * cell_to_leaf_c.T
        fine_to_fine = cell_to_leaf_f.T * cell_to_leaf_f
        # Fine cells are cells already fine:
        cells_ff = fine_to_fine * np.ones(self.level_grids[level+1].num_cells, dtype=bool)
        # of cells that should be refined
        cells_fc = to_refine * cells_leaf_to_ref > 0
        cells_f = cells_ff | cells_fc

        num_c_cells = np.sum(cells_c)
        num_f_cells = np.sum(cells_f)

        coarse_idx = np.where(cells_c)[0]
        indices = np.arange(num_c_cells)
        indptr = np.zeros(cells_c.size + 1, dtype=int)
        indptr[1:][cells_c] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        c2r = sps.csc_matrix((data, indices, indptr), shape=(num_c_cells, cells_c.size))

        fine_idx = np.where(cells_f)[0]
        indices = np.arange(num_f_cells)
        indptr = np.zeros(cells_f.size + 1 , dtype=int)
        indptr[1:][cells_f] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        f2r = sps.csc_matrix((data, indices, indptr), shape=(num_f_cells, cells_f.size))

        ###################
        face_to_leaf_c = self.project_level_to_leaf(level, 'face')

        faces_f = (np.abs(self.level_grids[level + 1].cell_faces) * cells_f) > 0
        coarse_to_coarse = face_to_leaf_c.T * face_to_leaf_c
        faces_c_to_c = coarse_to_coarse * np.ones(self.level_grids[level].num_faces, dtype=bool)
        faces_c = (proj_f * ~faces_f  > 0) & faces_c_to_c

        num_c_faces = np.sum(faces_c)
        num_f_faces = np.sum(faces_f)
        num_faces_new = num_c_faces + num_f_faces


        coarse_idx = np.where(faces_c)[0]
        indices = np.arange(num_c_faces)
        indptr = np.zeros(faces_c.size + 1 , dtype=int)
        indptr[1:][faces_c] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        cf2rf = sps.csc_matrix((data, indices, indptr), shape=(num_c_faces, faces_c.size))
        
        fine_idx = np.where(faces_f)[0]
        indices = np.arange(num_f_faces)
        indptr = np.zeros(faces_f.size + 1 , dtype=int)
        indptr[1:][faces_f] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        ff2rf = sps.csc_matrix((data, indices, indptr), shape=(num_f_faces, faces_f.size))

        # Map from fine faces to coarse cells
        # First find coarse faces that has been refined

        faces_fc = ((faces_f) * (proj_f.T  * np.abs(self.level_grids[level].cell_faces) * cells_c)) > 0
        mask = sps.diags(faces_f, dtype=bool)
        ff2c = ff2rf * mask

        ## Map nodes
        node_to_leaf_c = self.project_level_to_leaf(level, 'node')
        nodes_f = (np.abs(self.level_grids[level + 1].face_nodes) * faces_f) > 0
        coarse_to_coarse = node_to_leaf_c.T * node_to_leaf_c
        nodes_c_to_c = coarse_to_coarse * np.ones(self.level_grids[level].num_nodes, dtype=bool)
        nodes_c = (proj_n * ~nodes_f  > 0) & nodes_c_to_c

        num_c_nodes = np.sum(nodes_c)
        num_f_nodes = np.sum(nodes_f)
        num_nodes_new = num_c_nodes + num_f_nodes

        # Map from refined cells to coarse
        indices = np.arange(num_c_nodes)
        indptr = np.zeros(nodes_c.size + 1)
        indptr[1:][nodes_c] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        rn2cn = sps.csc_matrix((data, indices, indptr), shape=(num_c_nodes, nodes_c.size))

        # Map from refined cells to fine cells
        indices = np.arange(num_f_nodes)
        indptr = np.zeros(nodes_f.size + 1 , dtype=int)
        indptr[1:][nodes_f] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        rn2fn = sps.csc_matrix((data, indices, indptr), shape=(num_f_nodes, nodes_f.size))

        nodes_fc = ((nodes_f) * (proj_n.T * np.abs(self.level_grids[level].face_nodes) * faces_c)) > 0
        mask = sps.diags(nodes_f, dtype=bool)
        fn2c = rn2fn *  mask

        ##############
        # add cell_faces
        cell_faces_c = cf2rf * self.level_grids[level].cell_faces * c2r.T
        cell_faces_f = ff2rf * self.level_grids[level + 1].cell_faces * f2r.T
        cell_faces_cf = ff2c * proj_f.T * self.level_grids[level].cell_faces * c2r.T
#        cell_faces =  cell_faces_c + cell_faces_f + cell_faces_cf

        face_nodes_c = rn2cn * self.level_grids[level].face_nodes * cf2rf.T
        face_nodes_f = rn2fn * self.level_grids[level + 1].face_nodes * ff2rf.T
        face_nodes_cf = fn2c * proj_n.T * self.level_grids[level].face_nodes * cf2rf.T
        # Update grid
        self.block_cell_faces[level, level] = cell_faces_c
        self.block_cell_faces[level + 1, level] = cell_faces_cf
        self.block_cell_faces[level + 1, level + 1] = cell_faces_f

        if level > 0:
            proj = cf2rf * self.face_projections[level].T
            self.block_cell_faces[level, level - 1] =  proj * self.block_cell_faces[level, level - 1]

        if (level < len(self.level_grids) - 2 and
            not self.block_cell_faces[level + 2, level + 1] is None):

            proj = self.cell_projections[level + 1] * f2r.T
            new_copuling_cell_faces = self.block_cell_faces[level + 2, level + 1] * proj 
            self.block_cell_faces[level + 2, level + 1] = new_copuling_cell_faces

        self.block_face_nodes[level, level] = face_nodes_c
        self.block_face_nodes[level + 1, level] = face_nodes_cf
        self.block_face_nodes[level + 1, level + 1] = face_nodes_f
        if level > 0:
            proj = rn2cn * self.node_projections[level].T
            self.block_face_nodes[level, level - 1] = proj * self.block_face_nodes[level, level - 1]
        if (level < len(self.level_grids) - 2 and
            not self.block_face_nodes[level + 2, level + 1] is None):

            proj = self.face_projections[level + 1] * ff2rf.T
            new_copuling_face_nodes = self.block_face_nodes[level + 2, level + 1] * proj 
            self.block_face_nodes[level + 2, level + 1] = new_copuling_face_nodes


        self.block_nodes[level] = self.level_grids[level].nodes[:, nodes_c]
        self.block_face_normals[level] = self.level_grids[level].face_normals[:, faces_c]
        self.block_face_areas[level] = self.level_grids[level].face_areas[faces_c]
        self.block_cell_volumes[level] = self.level_grids[level].cell_volumes[cells_c]
        self.block_cell_centers[level] = self.level_grids[level].cell_centers[:, cells_c]

        self.block_nodes[level + 1] = self.level_grids[level + 1].nodes[:, nodes_f]
        self.block_face_normals[level + 1] = self.level_grids[level + 1].face_normals[:, faces_f]
        self.block_face_areas[level + 1] = self.level_grids[level + 1].face_areas[faces_f]
        self.block_cell_volumes[level + 1] = self.level_grids[level + 1].cell_volumes[cells_f]
        self.block_cell_centers[level + 1] = self.level_grids[level + 1].cell_centers[:, cells_f]

        self.cell_level = np.hstack([i * np.ones(cc.shape[1], dtype=int) for i, cc in enumerate(self.block_cell_centers) if cc is not None])
        self.num_cells = self.cell_level.size 
        self.num_faces = np.hstack(self.block_face_areas).size
        self.num_nodes = np.hstack(self.block_nodes).shape[1]

        
        self.cell_projections[level] = c2r#sps.vstack([c2r, zercc], dtype=bool)
        self.cell_projections[level + 1] = f2r#sps.vstack([zerfc, f2r], dtype=bool)

        self.face_projections[level] = cf2rf
        self.face_projections[level + 1] =  ff2rf

        self.node_projections[level] = rn2cn
        self.node_projections[level + 1] = rn2fn




    def update_grid_prop(self):
        self.cell_faces = sps.bmat(self.block_cell_faces, format='csc')
        self.face_nodes = sps.bmat(self.block_face_nodes, format='csc')
        self.nodes = np.hstack(self.block_nodes)
        self.face_normals = np.hstack(self.block_face_normals)
        self.face_areas = np.hstack(self.block_face_areas)
        self.cell_volumes = np.hstack(self.block_cell_volumes)
        self.cell_centers = np.hstack(self.block_cell_centers)
        self.num_faces = self.face_areas.size
        self.num_nodes = self.nodes.shape[1]



    def project_level_to_leaf(self, level, element_type):
        if element_type=="cell":
            active = self.cell_projections
            num_elements = self.level_grids[level].num_cells
        elif element_type=="face":
            active = self.face_projections
            num_elements = self.level_grids[level].num_faces
        elif element_type=="node":
            active = self.node_projections
            num_elements = self.level_grids[level].num_nodes
        else:
            raise ValueError(
                'Unknwon element_type: {}. Possible choises are: cell, face, node'.format(element_type)
            )

        block_mat = np.empty(len(self.level_grids), dtype=object)
        for i in range(len(self.level_grids)):
            if i==level:
                block_mat[i] = active[level]
            else:
                block_mat[i] = sps.csc_matrix((active[i].shape[0], num_elements))
        return sps.vstack(block_mat, format="csc", dtype=bool)

    def _init_active_cells(self):
        # Inital grid is the coarse grid:
        active_c = [None] * len(self.level_grids)
        active_c[0] = sps.diags(np.ones(self.num_cells, dtype=bool), dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            active_c[level + 1] = sps.csc_matrix((g.num_cells, g.num_cells), dtype=bool)
        self.active_cells = active_c

        active_f = [None] * len(self.level_grids)
        active_f[0] = sps.diags(np.ones(self.level_grids[0].num_faces, dtype=bool), dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            active_f[level + 1] = sps.csc_matrix((g.num_faces, g.num_faces), dtype=bool)
        self.active_faces = active_f

        active_n = [None] * len(self.level_grids)
        active_n[0] = sps.diags(np.ones(self.level_grids[0].num_nodes, dtype=bool), dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            active_n[level + 1] = sps.csc_matrix((g.num_nodes, g.num_nodes), dtype=bool)
        self.active_nodes = active_n


    def _init_projection(self):
        # Inital grid is the coarse grid:
        proj_c = [None] * len(self.level_grids)
        proj_c[0] = sps.diags(np.ones(self.num_cells, dtype=bool), dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            proj_c[level + 1] = sps.csc_matrix((0, g.num_cells))
        self.cell_projections = proj_c

        proj_f = [None] * len(self.level_grids)
        proj_f[0] = sps.diags(np.ones(self.level_grids[0].num_faces, dtype=bool), dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            proj_f[level + 1] = sps.csc_matrix((0, g.num_faces))
        self.face_projections = proj_f

        proj_n = [None] * len(self.level_grids)
        proj_n[0] = sps.diags(np.ones(self.level_grids[0].num_nodes, dtype=bool), dtype=bool)
        for level, g in enumerate(self.level_grids[1:]):
            proj_n[level + 1] = sps.csc_matrix((0, g.num_nodes))
        self.node_projections = proj_n


if __name__=="__main__":
    import time
    import matplotlib.pyplot as plt

    tic = time.time()
    lg = CartLeafGrid([2, 2], [1, 1], 3)
    print("time to generate leaf grid: {} s".format(time.time() - tic))

    tic = time.time()
    lg.refine_cells(0)
    lg.refine_cells(3)
    lg.refine_cells(0)

    lg.compute_geometry()
    print("time to refine leaf grid: {} s".format(time.time() - tic))

    plt.plot(lg.cell_centers[0], lg.cell_centers[1], '.')
    plt.plot(lg.nodes[0], lg.nodes[1], 'o')
    plt.plot(lg.face_centers[0], lg.face_centers[1], 'x')

    plt.show()
    pp.plot_grid(lg)
