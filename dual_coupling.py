import numpy as np
import scipy.sparse as sps

class DualCoupling(object):

#------------------------------------------------------------------------------#

    def __init__(self, solver):
        self.solver = solver

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, face_cells, data):

        k = data['kn']
        dof = np.array([self.solver.ndof(g_h), self.solver.ndof(g_l)])

        cells_l, faces_h, _ = sps.find(face_cells)
        faces, _, sgn = sps.find(g_h.cell_faces)
        sgn = sgn[np.unique(faces, return_index=True)[1]]
        sgn = sgn[faces_h]

        cc = np.array([sps.coo_matrix((i,j)) for i in dof for j in dof]\
                     ).reshape((2, 2))

        dataIJ, I, J = sgn, g_l.num_faces+cells_l, faces_h
        cc[1,0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0,1] = cc[1,0].T

        dataIJ = 1./np.multiply(g_h.face_areas[faces_h], k[cells_l])
        I, J = faces_h, faces_h
        cc[0,0] = sps.csr_matrix((dataIJ, (I, J)), (dof[0], dof[0]))

        return cc

#------------------------------------------------------------------------------#
