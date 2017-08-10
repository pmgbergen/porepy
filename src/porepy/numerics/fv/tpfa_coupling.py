import copy
import numpy as np
import scipy.sparse as sps
from porepy.utils.comp_geom import map_grid
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling


class TpfaCoupling(AbstractCoupling):

    def __init__(self, solver):
        self.solver = solver

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Computes the coupling terms for the faces between cells in g_h and g_l
        using the two-printoint flux approximation.

        Parameters:
            g_h and g_l: grid structures of the higher and lower dimensional
                subdomains, respectively.
            data_h and data_l: the corresponding data dictionaries. Assumed
                to contain both permeability values ('perm') and apertures
                ('apertures') for each of the cells in the grids.

        Returns:
            cc: Discretization matrices for the coupling terms assembled
                in a csc.sparse matrix.
        """

        k_l = data_l['param'].get_tensor(self.solver)
        k_h = data_h['param'].get_tensor(self.solver)
        a_l = data_l['param'].get_aperture()
        a_h = data_h['param'].get_aperture()

        dof = np.array([self.solver.ndof(g_h), self.solver.ndof(g_l)])

        # Obtain the cells and face signs of the higher dimensional grid
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        faces, cells_h, sgn_h = sps.find(g_h.cell_faces)
        ind = np.unique(faces, return_index=True)[1]
        sgn_h = sgn_h[ind]
        cells_h = cells_h[ind]

        cells_h, sgn_h = cells_h[faces_h], sgn_h[faces_h]

        # The procedure for obtaining the face transmissibilities of the higher
        # grid is analougous to the one used in numerics.fv.tpfa.py, see that file
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
        t_face_h = t_face_h * a_h[cells_h]
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
        t_face_l = a_h[cells_h] * normal_perm

        # And use it for face-center cell-center distance
        t_face_l = np.divide(
            t_face_l, 0.5 * np.divide(a_l[cells_l], a_h[cells_h]))

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

        # Save the flux discretization for back-computation of fluxes
        cells2faces = sps.csr_matrix((sgn_h, (faces_h, cells_h)),
                                     (g_h.num_faces, g_h.num_cells))

        data_edge['coupling_flux'] = sps.hstack([cells2faces * cc[0, 0],
                                                 cells2faces * cc[0, 1]])
        data_edge['coupling_discretization'] = cc

        return cc

    #------------------------------------------------------------------------------#


    def compute_discharges(self, gb):
        """
        Computes discharges over all faces in the entire grid bucket given
        pressures for all nodes, provided as node properties.

        Parameter:
            gb: grid bucket with the following data fields for all nodes/grids:
                    'flux': Internal discretization of fluxes.
                    'bound_flux': Discretization of boundary fluxes.
                    'p': Pressure values for each cell of the grid.
                    'bc_val': Boundary condition values.
                and the following edge property field for all connected grids:
                    'coupling_flux': Discretization of the coupling fluxes.
        Returns:
            gb, the same grid bucket with the added field 'discharge' added to all
            node data fields. Note that the fluxes between grids will be added doubly,
            both to the data corresponding to the higher dimensional grid and as a
            edge property. This is not true in the case of edges between grids of equal
            dimension. There, the fluxes are only added at the edge.
        """
        
        

        for gr, da in gb:
            if gr.dim > 0:
                pa = da['param']
                f, _, s = sps.find(gr.cell_faces)
                _, ind = np.unique(f, return_index=True)
                s = s[ind]
                fl = da['flux']
               
                dis = fl * da['p'] + da['bound_flux'] \
                                   * pa.get_bc_val(self.solver)
                
                pa.set_discharge(dis)

        
        for e, data in gb.edges_props():
            # According to the sorting convention, g2 is the higher dimensional grid,
            # the one to who's faces the fluxes correspond 
            g1, g2 = gb.sorted_nodes_of_edge(e)
            
            if data['face_cells'] is not None and g1.dim != g2.dim:
                pa = data['param']#
                coupling_flux = gb.edge_prop(e, 'coupling_flux')[0]
                pressures = gb.nodes_prop([g2, g1], 'p')
                coupling_contribution = coupling_flux * np.concatenate(pressures)
                pa_2 = gb.node_prop(g2, 'param')
                flux = coupling_contribution + pa_2.get_discharge()
                pa_2.set_discharge(copy.deepcopy(flux))                
                pa.set_discharge(flux)
                
            
            elif g1.dim == g2.dim:
                pa = data['param']
                # g2 is now only the "higher", but still the one defining the faces
                # (cell-cells connections) in the sense that the normals are assumed
                # outward from g2, "pointing towards the g1 cells". Note that in
                # general, there are g2.num_cells x g1.num_cells possible connections.
                # In the 1d-1d case, only one connection is active.
                        
                coupling_flux = gb.edge_prop(e, 'coupling_flux')[0].todense()
                pressures = gb.nodes_prop([g2, g1], 'p')
                
                # The dense-sparse conversion may be avoidable if einsum is
                # removed
                contribution_0 = np.einsum('ij,i->ij',
                                           coupling_flux,pressures[0])
                contribution_1 = np.einsum('ij,j->ij',
                                           coupling_flux,pressures[1])
                flux = contribution_0-contribution_1
                # Store flux at the edge only. This means that the flux will remain
                # zero in the data of both g1 and g2
                
                pa.set_discharge(flux)
            
        
    #------------------------------------------------------------------------------#
