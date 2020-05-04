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
            self.mesh_sizes.append( np.asarray(nx) * (level + 1) )
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

        self.cell_level = np.zeros(self.num_cells, dtype=int)


    def cell_projection(self, level0, level1):
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

    def face_projection(self, level0, level1):
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


        nx0 = self.mesh_sizes[level0] + 1
        nx1 = self.mesh_sizes[level1] + 1

        offset_x = np.atleast_2d(np.cumsum([nx0[0]] * (nx0[1] - 1))).T
        offset_x -= offset_x[0]
        face_indices_x = np.tile(np.tile(np.arange(nx0[0]), (1, 2)).ravel('F'), (nx0[1] - 1, 1))
        face_indices_x += offset_x
        face_indices_x = face_indices_x.ravel('C')

        offset_y = np.atleast_2d(np.cumsum([nx0[1]] * (nx0[0] - 1))).T
        offset_y -= offset_y[0]
        offset_y += (nx0[0] * (nx0[1] - 1))
        
        face_indices_y = np.tile(np.tile(np.arange(nx0[0] - 1), (2, 1)).ravel('F'), (nx0[1], 1))
        face_indices_y += offset_y
        face_indices_y = face_indices_y.ravel('C')

        face_indices = np.hstack((face_indices_x, face_indices_y))

        face_ptr = np.arange(g1.num_faces + 1)

        maskx = np.ones(nx1[0], dtype=bool)
        maskx[1::2] = False
        maskx = np.tile(maskx, (1, nx1[1] -1)).ravel()

        masky = np.vstack((np.ones(nx1[0] -1, dtype=bool), np.zeros(nx1[0] - 1, dtype=bool)))
        masky = np.tile(masky, (nx0[1], 1))
        masky = masky[:-1].ravel('C')
        mask = np.hstack((maskx, masky)).ravel()

        mask_padded = np.hstack((True, mask))
        offset = np.cumsum(~mask_padded)
        face_ptr -= offset
        data = np.ones(face_indices.size, dtype=int)

        return sps.csc_matrix((data, face_indices, face_ptr))

    def node_projection(self, level0, level1):
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
        if not isinstance(np.asarray(cells).dtype, bool):
            cells_c = np.ones(self.num_cells, dtype=bool)
            cells_c[cells] = False
        else:
            cells_c = cells


        proj_c = self.cell_projection(0, 1)
        proj_f = self.face_projection(0, 1)
        proj_n = self.node_projection(0, 1)

        cells_f = (proj_c.T * ~cells_c) > 0

        num_c_cells = np.sum(cells_c)
        num_f_cells = np.sum(cells_f)
        num_cells_new = num_c_cells + num_f_cells

        coarse_idx = np.where(cells_c)[0]
        indices = np.arange(num_c_cells)
        indptr = np.zeros(cells_c.size + 1 , dtype=int)
        indptr[coarse_idx + 1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        c2r = sps.csc_matrix((data, indices, indptr), shape=(num_cells_new, cells_c.size))

        fine_idx = np.where(cells_f)[0]
        indices = np.arange(num_c_cells, num_cells_new)
        indptr = np.zeros(cells_f.size + 1 , dtype=int)
        indptr[fine_idx + 1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        f2r = sps.csc_matrix((data, indices, indptr), shape=(num_cells_new, cells_f.size))
        
        ###################
        faces_f = (np.abs(self.level_grids[1].cell_faces) * cells_f) > 0
        faces_c = (proj_f * ~faces_f ) > 0 

        num_c_faces = np.sum(faces_c)
        num_f_faces = np.sum(faces_f)
        num_faces_new = num_c_faces + num_f_faces

        coarse_idx = np.where(faces_c)[0]
        indices = np.arange(num_c_faces)
        indptr = np.zeros(faces_c.size + 1 , dtype=int)
        indptr[coarse_idx + 1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        cf2rf = sps.csc_matrix((data, indices, indptr), shape=(num_faces_new, faces_c.size))
        
        fine_idx = np.where(faces_f)[0]
        indices = np.arange(num_c_faces, num_faces_new)
        indptr = np.zeros(faces_f.size + 1 , dtype=int)
        indptr[fine_idx + 1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        ff2rf = sps.csc_matrix((data, indices, indptr), shape=(num_faces_new, faces_f.size))

        # Map from refined cells to coarse
        indices = np.where(faces_c)[0]
        indptr = np.zeros(num_faces_new + 1 , dtype=int)
        indptr[1:num_c_faces + 1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        rf2cf = sps.csc_matrix((data, indices, indptr), shape=(faces_c.size, num_faces_new))

        # Map from refined cells to fine cells
        indices = np.where(faces_f)[0]
        indptr = np.zeros(num_faces_new + 1 , dtype=int)
        indptr[num_c_faces + 1:num_faces_new + 1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        rf2ff = sps.csc_matrix((data, indices, indptr), shape=(faces_f.size, num_faces_new))


        # Map from fine faces to coarse cells
        # First find coarse faces that has been refined
        faces_fc = ((faces_f) * (proj_f.T * np.abs(self.level_grids[0].cell_faces) * cells_c)) > 0
        mask = sps.diags(faces_f, dtype=bool)
        ff2c = rf2ff.T * mask

        ## Map nodes
        nodes_f = (np.abs(self.level_grids[1].face_nodes) * faces_f) > 0
        nodes_c = (proj_n * ~nodes_f ) > 0 

        num_c_nodes = np.sum(nodes_c)
        num_f_nodes = np.sum(nodes_f)
        num_nodes_new = num_c_nodes + num_f_nodes

        # Map from refined cells to coarse
        indices = np.where(nodes_c)[0]
        indptr = np.zeros(num_nodes_new + 1 , dtype=int)
        indptr[1:num_c_nodes + 1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        rn2cn = sps.csc_matrix((data, indices, indptr), shape=(nodes_c.size, num_nodes_new))

        # Map from refined cells to fine cells
        indices = np.where(nodes_f)[0]
        indptr = np.zeros(num_nodes_new + 1 , dtype=int)
        indptr[num_c_nodes + 1:num_nodes_new+1] = 1
        indptr = np.cumsum(indptr)
        data = np.ones(indices.size, dtype=bool)
        rn2fn = sps.csc_matrix((data, indices, indptr), shape=(nodes_f.size, num_nodes_new))

        nodes_fc = ((nodes_f) * (proj_n.T * np.abs(self.level_grids[0].face_nodes) * faces_c)) > 0
        mask = sps.diags(nodes_f, dtype=bool)
        fn2c = rn2fn.T * mask

        ##############
        # add cell_faces
        cell_faces_c = rf2cf.T * self.level_grids[0].cell_faces * c2r.T
        cell_faces_f = rf2ff.T * self.level_grids[1].cell_faces * f2r.T
        cell_faces_cf = ff2c * proj_f.T * self.level_grids[0].cell_faces * c2r.T
        cell_faces =  cell_faces_c + cell_faces_f + cell_faces_cf

        face_nodes_c = rn2cn.T * self.level_grids[0].face_nodes * cf2rf.T
        face_nodes_f = rn2fn.T * self.level_grids[1].face_nodes * ff2rf.T
        face_nodes_cf = fn2c * proj_n.T * self.level_grids[0].face_nodes * cf2rf.T
        face_nodes =  face_nodes_c + face_nodes_f + face_nodes_cf

        # Update grid
        self.num_cells = num_cells_new
        self.num_faces= num_faces_new
        self.num_nodes = num_nodes_new
        self.cell_centers = np.hstack((self.level_grids[0].cell_centers[:, cells_c],
                                   self.level_grids[1].cell_centers[:, cells_f]))


        self.face_centers = np.hstack((self.level_grids[0].face_centers[:, faces_c],
                                   self.level_grids[1].face_centers[:, faces_f]))
        self.face_normals = np.hstack((self.level_grids[0].face_normals[:, faces_c],
                                   self.level_grids[1].face_normals[:, faces_f]))
        self.nodes = np.hstack((self.level_grids[0].nodes[:, nodes_c],
                                self.level_grids[1].nodes[:, nodes_f]))

        self.cell_faces = cell_faces.tocsc()
        self.face_nodes = face_nodes.tocsc()



# lg = CartLeafGrid([2, 2], [1, 1], 2)
# g0 = lg.level_grids[0]
# g1 = lg.level_grids[1]


# proj_cc = lg.cell_projection(0, 1)
# proj_ff = lg.face_projection(0, 1)

# lg.refine_cells(0)

# import matplotlib.pyplot as plt
# lg.compute_geometry()
# plt.plot(lg.cell_centers[0], lg.cell_centers[1], '.')
# plt.plot(lg.face_centers[0], lg.face_centers[1], 'o')
# plt.plot(lg.nodes[0], lg.nodes[1], 'x')

# for f in range(lg.num_faces):
#     node0 = lg.face_nodes.indices[lg.face_nodes.indptr[f]]
#     node1 = lg.face_nodes.indices[lg.face_nodes.indptr[f + 1]-1]
#     dx = lg.nodes[0, node1] - lg.nodes[0, node0]
#     dy = lg.nodes[1, node1] - lg.nodes[1, node0]
#     plt.arrow(lg.nodes[0, node0], lg.nodes[1, node0], dx, dy, head_width=.06)
# plt.axis('equal')
# plt.xlim([-0.5, 1.5])
# plt.ylim([-0.5, 1.5])
# plt.show()

# pp.plot_grid(lg)
