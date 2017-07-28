import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling


class UpwindCoupling(AbstractCoupling):

#------------------------------------------------------------------------------#

    def __init__(self, solver):
        self.solver = solver

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Construct the matrix (and right-hand side) for the coupling conditions.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: dictionary which stores the data for the higher dimensional
                grid
            data_l: dictionary which stores the data for the lower dimensional
                grid
            data: dictionary which stores the data for the edges of the grid
                bucket

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.
        """

        # Normal component of the velocity from the higher dimensional grid
        discharge = data_edge['param'].get_discharge()

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # 1d-1d
        if g_h.dim == g_l.dim:
            # Remember that face_cells are really cell-cell connections
            # in this case
            cells_h, cells_l,_ = sps.find(data_edge['face_cells'])
            diag_cc11 = np.zeros(g_l.num_cells)
            diag_cc00 = np.zeros(g_h.num_cells)
            # Need only one discharge, from h to l. Positive values
            # go from h to l.
            
            
            d_00 = (np.sign(discharge.clip(min=0)) * discharge).sum(axis=1)
            d_11 = (np.sign(discharge.clip(max=0)) * discharge).sum(axis=0)
            
            
            np.add.at(diag_cc00, range(g_h.num_cells), d_00)
            np.add.at(diag_cc11, range(g_l.num_cells), d_11)
            # Compute the outflow from the second to the first grid
            cc[1, 0] = sps.coo_matrix(-discharge.clip(min=0).T)
            # Compute the inflow to the first from the second grid
            cc[0, 1] = sps.coo_matrix(discharge.clip(max=0))    
            
        else:
            # Recover the information for the grid-grid mapping
            cells_l, faces_h, _ = sps.find(data_edge['face_cells'])

            # Recover the correct sign for the velocity
            faces, _, sgn = sps.find(g_h.cell_faces)
            sgn = sgn[np.unique(faces, return_index=True)[1]]
            discharge = sgn[faces_h] * discharge[faces_h]

            # Determine which are the corresponding cells of the faces_h
            cell_faces_h = g_h.cell_faces.tocsr()[faces_h, :]
            cells_h = cell_faces_h.nonzero()[1]

            diag_cc11 = np.zeros(g_l.num_cells)
            np.add.at(diag_cc11, cells_l, np.sign(discharge.clip(max=0)) * discharge)

            diag_cc00 = np.zeros(g_h.num_cells)
            np.add.at(diag_cc00, cells_h, np.sign(discharge.clip(min=0)) * discharge)
            # Compute the outflow from the higher to the lower dimensional grid
            cc[1, 0] = sps.coo_matrix((-discharge.clip(min=0), (cells_l, cells_h)),
                                      shape=(dof[1], dof[0]))

            # Compute the inflow from the higher to the lower dimensional grid
            cc[0, 1] = sps.coo_matrix((discharge.clip(max=0), (cells_h, cells_l)),
                                      shape=(dof[0], dof[1]))

        cc[1, 1] = sps.dia_matrix((diag_cc11, 0), shape=(dof[1], dof[1]))

        cc[0, 0] = sps.dia_matrix((diag_cc00, 0), shape=(dof[0], dof[0]))
       
        return cc

#------------------------------------------------------------------------------#

    def cfl(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Return the time step according to the CFL condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: dictionary which stores the data for the higher dimensional
                grid
            data_l: dictionary which stores the data for the lower dimensional
                grid
            data: dictionary which stores the data for the edges of the grid
                bucket

        Return:
            deltaT: time step according to CFL condition.

        """
        # Retrieve the discharge, which is mandatory
        discharge = data_edge['param'].get_discharge()
        aperture_h = data_h['param'].get_aperture()
        aperture_l = data_l['param'].get_aperture()
        phi_l = data_l['param'].get_porosity()
        if g_h.dim==g_l.dim:
            return np.inf#########
            cells_h, cells_l,_ = sps.find(data_edge['face_cells'])
            not_zero = ~np.isclose(np.zeros(discharge.size), discharge, atol = 0)
            if not np.any(not_zero):
                return np.Inf
            
        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])

        # Detect and remove the faces which have zero in "discharge"
        not_zero = ~np.isclose(np.zeros(faces_h.size), discharge[faces_h], atol=0)
        if not np.any(not_zero):
            return np.inf

        cells_l = cells_l[not_zero]
        faces_h = faces_h[not_zero]
        # Mapping from faces_h to cell_h
        
        cell_faces_h = g_h.cell_faces.tocsr()[faces_h, :]
        print('cfh', cell_faces_h)
        print(cell_faces_h.nonzero()[1], not_zero)
        cells_h = cell_faces_h.nonzero()[1][not_zero]
        # Retrieve and map additional data
        aperture_h = aperture_h[cells_h]
        aperture_l = aperture_l[cells_l]
        phi_l = phi_l[cells_l]
        # Compute discrete distance cell to face centers for the lower
        # dimensional grid
        dist = 0.5 * np.divide(aperture_l, aperture_h)
        # Since discharge is multiplied by the aperture, we get rid of it!!!!
        discharge = np.divide(discharge[faces_h],
                              g_h.face_areas[faces_h]*aperture_h)
        # deltaT is deltaX/discharge with coefficient
        return np.amin(np.abs(np.divide(dist, discharge)) * phi_l)

#------------------------------------------------------------------------------#
