"""
Module of coupling laws for hyperbolic equations.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.numerics.interface_laws.abstract_interface_law


class UpwindCoupling(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    def __init__(self, keyword):
        super(UpwindCoupling, self).__init__(keyword)

    def key(self):
        return self.keyword + "_"

    def discretization_key(self):
        return self.key() + pp.DISCRETIZATION

    def ndof(self, mg):
        return mg.num_cells

    def discretize(
        self, g_primary, g_secondary, data_primary, data_secondary, data_edge
    ):
        pass

    def assemble_matrix_rhs(
        self, g_primary, g_secondary, data_primary, data_secondary, data_edge, matrix
    ):
        """
        Construct the matrix (and right-hand side) for the coupling conditions.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_primary: grid of higher dimension
            g_secondary: grid of lower dimension
            data_primary: dictionary which stores the data for the higher dimensional
                grid
            data_secondary: dictionary which stores the data for the lower dimensional
                grid
            data_edge: dictionary which stores the data for the edges of the grid
                bucket
            matrix: Uncoupled discretization matrix.

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.

        """

        # Normal component of the velocity from the higher dimensional grid

        # @ALL: This should perhaps be defined by a globalized keyword
        lam_flux = data_edge[pp.PARAMETERS][self.keyword]["darcy_flux"]
        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        mg = data_edge["mortar_grid"]

        # We know the number of dofs from the primary and secondary side from their
        # discretizations
        dof = np.array([matrix[0, 0].shape[1], matrix[1, 1].shape[1], mg.num_cells])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # Projection from mortar to upper dimenional faces

        # mapping from upper dim cellls to faces
        # The mortars always points from upper to lower, so we don't flip any
        # signs
        face_to_cell_h = np.abs(pp.fvutils.scalar_divergence(g_primary))
        # We also need a trace-like projection from cells to faces
        trace_h = face_to_cell_h.T

        # Find upwind weighting. if flag is True we use the upper weights
        # if flag is False we use the lower weighs
        flag = (lam_flux > 0).astype(np.float)
        not_flag = 1 - flag

        # assemble matrices
        # Note the sign convention: The Darcy mortar flux is positive if it goes
        # from g_h to g_l. Thus a positive transport flux (assuming positive
        # concentration) will go out of g_h, into g_l.

        # Transport out of upper equals lambda.
        # Use integrated projcetion operator; the flux is an extensive quantity
        cc[0, 2] = face_to_cell_h * mg.mortar_to_primary_int()

        # transport out of lower is -lambda
        cc[1, 2] = -mg.mortar_to_secondary_int()

        # Discretisation of mortars
        # If fluid flux(lam_flux) is positive we use the upper value as weight,
        # i.e., T_primaryat * fluid_flux = lambda.
        # We set cc[2, 0] = T_primaryat * fluid_flux
        # Use averaged projection operator for an intensive quantity
        cc[2, 0] = sps.diags(lam_flux * flag) * mg.primary_to_mortar_avg() * trace_h

        # If fluid flux is negative we use the lower value as weight,
        # i.e., T_check * fluid_flux = lambda.
        # we set cc[2, 1] = T_check * fluid_flux
        # Use averaged projection operator for an intensive quantity
        cc[2, 1] = sps.diags(lam_flux * not_flag) * mg.secondary_to_mortar_avg()

        # The rhs of T * fluid_flux = lambda
        # Recover the information for the grid-grid mapping
        cc[2, 2] = -sps.eye(mg.num_cells)

        if data_primary["node_number"] == data_secondary["node_number"]:
            # All contributions to be returned to the same block of the
            # global matrix in this case
            cc = np.array([np.sum(cc, axis=(0, 1))])

        # rhs is zero
        rhs = np.squeeze([np.zeros(dof[0]), np.zeros(dof[1]), np.zeros(dof[2])])
        matrix += cc
        return matrix, rhs

    def cfl(
        self,
        g_primary,
        g_secondary,
        data_primary,
        data_secondary,
        data_edge,
        d_name="mortar_solution",
    ):
        """
        Return the time step according to the CFL condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        darcy_flux : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters:
            g_primary: grid of higher dimension
            g_secondary: grid of lower dimension
            data_primary: dictionary which stores the data for the higher dimensional
                grid
            data_secondary: dictionary which stores the data for the lower dimensional
                grid
            data: dictionary which stores the data for the edges of the grid
                bucket

        Return:
            deltaT: time step according to CFL condition.

        Note: the design of this function has not been updated according
        to the mortar structure. Instead, mg.high_to_mortar_int.nonzero()[1]
        is used to map the 'mortar_solution' (one flux for each mortar dof) to
        the old darcy_flux (one flux for each g_primary face).

        """
        # Retrieve the darcy_flux, which is mandatory

        aperture_primary = data_primary["param"].get_aperture()
        aperture_secondary = data_secondary["param"].get_aperture()
        phi_secondary = data_secondary["param"].get_porosity()
        mg = data_edge["mortar_grid"]
        darcy_flux = np.zeros(g_primary.num_faces)
        darcy_flux[mg.high_to_mortar_int.nonzero()[1]] = data_edge[d_name]
        if g_primary.dim == g_secondary.dim:
            # More or less same as below, except we have cell_cells in the place
            # of face_cells (see grid_bucket.duplicate_without_dimension).
            phi_primary = data_primary["param"].get_porosity()
            cells_secondary, cells_primary = data_edge["face_cells"].nonzero()
            not_zero = ~np.isclose(np.zeros(darcy_flux.shape), darcy_flux, atol=0)
            if not np.any(not_zero):
                return np.Inf

            diff = (
                g_primary.cell_centers[:, cells_primary]
                - g_secondary.cell_centers[:, cells_secondary]
            )
            dist = np.linalg.norm(diff, 2, axis=0)

            # Use minimum of cell values for convenience
            phi_secondary = phi_secondary[cells_secondary]
            phi_primary = phi_primary[cells_primary]
            apt_primary = aperture_primary[cells_primary]
            apt_secondary = aperture_secondary[cells_secondary]
            coeff = np.minimum(phi_primary, phi_secondary) * np.minimum(
                apt_primary, apt_secondary
            )
            return np.amin(np.abs(np.divide(dist, darcy_flux)) * coeff)

        # Recover the information for the grid-grid mapping
        cells_secondary, faces_primary, _ = sps.find(data_edge["face_cells"])

        # Detect and remove the faces which have zero in "darcy_flux"
        not_zero = ~np.isclose(
            np.zeros(faces_primary.size), darcy_flux[faces_primary], atol=0
        )
        if not np.any(not_zero):
            return np.inf

        cells_secondary = cells_secondary[not_zero]
        faces_primary = faces_primary[not_zero]
        # Mapping from faces_primary to cell_primary
        cell_faces_primary = g_primary.cell_faces.tocsr()[faces_primary, :]
        cells_primary = cell_faces_primary.nonzero()[1][not_zero]
        # Retrieve and map additional data
        aperture_primary = aperture_primary[cells_primary]
        aperture_secondary = aperture_secondary[cells_secondary]
        phi_secondary = phi_secondary[cells_secondary]
        # Compute discrete distance cell to face centers for the lower
        # dimensional grid
        dist = 0.5 * np.divide(aperture_secondary, aperture_primary)
        # Since darcy_flux is multiplied by the aperture wighted face areas, we
        # divide through that quantity to get velocities in [length/time]
        velocity = np.divide(
            darcy_flux[faces_primary],
            g_primary.face_areas[faces_primary] * aperture_primary,
        )
        # deltaT is deltaX/velocity with coefficient
        return np.amin(np.abs(np.divide(dist, velocity)) * phi_secondary)
