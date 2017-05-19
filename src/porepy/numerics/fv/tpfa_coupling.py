import numpy as np
import scipy.sparse as sps
from porepy.utils.comp_geom import map_grid


class TpfaCoupling(object):

    #------------------------------------------------------------------------------#

    def __init__(self, solver):
        self.solver = solver

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Computes the coupling terms for the faces between cells in g_h and g_l 
        using the two-printoint flux approximation.  

        Parameters:
            g_h and g_l: grid structures of the higher and lower dimensional 
                subdomains, respectively. 
            data_h and data_l: the corresponding data dictionaries. Assumed 
                to contain both permeability values ('k') and apertures ('a')
                for each of the cells in the grids.

        Returns:
            cc: Discretization matrices for the coupling terms assembled 
                in a csc.sparse matrix.
        """

        k_l = data_l['k']
        k_h = data_h['k']
        a_l = data_l['a']
        a_h = data_h['a']
        dof = np.array([self.solver.ndof(g_h), self.solver.ndof(g_l)])

        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])

        # Obtain the cells and face signs of the higher dimensional grid
        faces, cells_h, sgn_h = sps.find(g_h.cell_faces)
        ind = np.unique(faces, return_index=True)[1]
        sgn_h = sgn_h[ind]
        cells_h = cells_h[ind]

        cells_h, sgn_h = cells_h[faces_h], sgn_h[faces_h]

        # The procedure for obtaining the face transmissibilities of the higher
        # grid is analougous to the one used in fvdiscr.tpfa.py, see that file
        # for explanations
        n = g_h.face_normals[:, faces_h]
        n *= sgn_h
        perm_h = k_h.perm[:, :, cells_h]

        fc_cc_h = g_h.face_centers[::, faces_h] - g_h.cell_centers[::, cells_h]
        nk_h = perm_h * n

        nk_h = nk_h.sum(axis=0)
        nk_h *= fc_cc_h
        t_face_h = nk_h.sum(axis=0)

        # Account for the apertures
        t_face_h = t_face_h * np.power(a_h[cells_h], 3 - g_h.dim)

        dist_face_cell_h = np.power(fc_cc_h, 2).sum(axis=0)
        t_face_h = np.divide(t_face_h, dist_face_cell_h)

        # For the lower dimension some simplifications can be made, due to the
        # alignment of the face normals and (normal) permeabilities of the
        # cells. First, the normal component of the permeability of the lower
        # dimensional cells must be found. While not provided in g_l, the
        # normal of these faces is the same as that of the corresponding higher
        # dimensional face, up to a sign.
        n1 = n[np.newaxis, :, :]
        n2 = n[:, np.newaxis, :]
        n1n2 = n1 * n2

        normal_perm = np.einsum(
            'ij...,ij...', n1n2, k_l.perm[:, :, cells_l])
        # The area has been multiplied in twice, not once as above, through n1
        # and n2
        normal_perm = np.divide(normal_perm, g_h.face_areas[faces_h])

        # Account for aperture contribution to face area
        t_face_l = np.power(a_l[cells_l], 3 - g_h.dim) * normal_perm

        # And use it for face-center cell-center distance
        t_face_l = np.divide(t_face_l, 0.5 * a_l[cells_l])

        # Assemble face transmissibilities for the two dimensions and compute
        # harmonic average
        t_face = np.array([t_face_h, t_face_l])
        t = t_face.prod(axis=0) / t_face.sum(axis=0)

        # Create the block matrix for the contributions
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof]
                      ).reshape((2, 2))

        # Compute the off-diagonal terms
        dataIJ, I, J = -t, cells_l, cells_h
        cc[1, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0, 1] = cc[1, 0].T

        # Compute the diagonal terms
        dataIJ, I, J = t, cells_h, cells_h
        cc[0, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[0], dof[0]))
        I, J = cells_l, cells_l
        cc[1, 1] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[1]))
        return cc

#------------------------------------------------------------------------------#
