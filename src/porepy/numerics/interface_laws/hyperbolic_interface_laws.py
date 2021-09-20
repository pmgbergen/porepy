"""
Module of coupling laws for hyperbolic equations.
"""
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.numerics.interface_laws.abstract_interface_law

module_sections = ["numerics"]


class UpwindCoupling(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)

        # Keywords for accessing discretization matices

        # Trace operator for the primary grid
        self.trace_primary_matrix_key = "trace"
        # Inverse trace operator (face -> cell)
        self.inv_trace_primary_matrix_key = "inv_trace"
        # Matrix for filtering upwind values from the primary grid
        self.upwind_primary_matrix_key = "upwind_primary"
        # Matrix for filtering upwind values from the secondary grid
        self.upwind_secondary_matrix_key = "upwind_secondary"
        # Matrix that carries the fluxes
        self.flux_matrix_key = "flux"
        # Discretization of the mortar variable
        self.mortar_discr_matrix_key = "mortar_discr"

    def key(self) -> str:
        return self.keyword + "_"

    def discretization_key(self):
        return self.key() + pp.DISCRETIZATION

    def ndof(self, mg: pp.MortarGrid) -> int:
        return mg.num_cells

    @pp.time_logger(sections=module_sections)
    def discretize(
        self,
        g_primary: pp.Grid,
        g_secondary: pp.Grid,
        data_primary: Dict,
        data_secondary: Dict,
        data_edge: Dict,
    ) -> None:

        # First check if the grid dimensions are compatible with the implementation.
        # It is not difficult to cover the case of equal dimensions, it will require
        # trace operators for both grids, but it has not yet been done.
        if g_primary.dim != g_secondary.dim + 1:
            raise ValueError(
                "Implementation is only valid for grids one dimension apart."
            )

        matrix_dictionary = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        
        # Number of aqueous components. Defualt is 1 
        num_components: int = data_edge[pp.PARAMETERS][self.keyword].get("num_components", 1)
        eye_num_comp = sps.eye(num_components)
  
        # Normal component of the velocity from the higher dimensional grid
        lam_flux: np.ndarray = data_edge[pp.PARAMETERS][self.keyword]["darcy_flux"]
        
        mg: pp.MortarGrid = data_edge["mortar_grid"]

        # mapping from upper dim cellls to faces
        # The mortars always points from upper to lower, so we don't flip any
        # signs.
        # The mapping will be non-zero also for faces not adjacent to
        # the mortar grid, however, we wil hit it with mortar projections, thus kill
        # those elements
        inv_trace_h = np.abs(pp.fvutils.scalar_divergence(g_primary))
        # We also need a trace-like projection from cells to faces
        trace_h = inv_trace_h.T
        
        # Find upwind weighting. if flag is True we use the upper weights
        # if flag is False we use the lower weighs
        flag = (lam_flux > 0).astype(float)
        not_flag = 1 - flag

        # Discretizations are the flux, but masked so that only the upstream direction
        # is hit.
        upwind_from_primary = sps.diags(flag)
        upwind_from_secondary = sps.diags(not_flag)
        flux = sps.diags(lam_flux)

        # Extend all the matrices, so they take several components into account
        matrix_dictionary[self.inv_trace_primary_matrix_key] = sps.kron(
        inv_trace_h, eye_num_comp 
        ).tocsr()  
        
        matrix_dictionary[self.trace_primary_matrix_key] = sps.kron(
        trace_h, eye_num_comp 
        ).tocsr()
    
        matrix_dictionary[self.upwind_primary_matrix_key] = sps.kron(
            upwind_from_primary, eye_num_comp 
            ).tocsr()
        
        matrix_dictionary[self.upwind_secondary_matrix_key] = sps.kron(
            upwind_from_secondary, eye_num_comp 
            ).tocsr()

        matrix_dictionary[self.flux_matrix_key] = sps.kron(
            flux, eye_num_comp 
            ).tocsr()

        # Identity matrix, to represent the mortar variable itself
        matrix_dictionary[self.mortar_discr_matrix_key] = sps.kron(
            sps.eye(mg.num_cells), eye_num_comp 
            ).tocsr()

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(
        self,
        g_primary: pp.Grid,
        g_secondary: pp.Grid,
        data_primary: Dict,
        data_secondary: Dict,
        data_edge,
        matrix: sps.spmatrix,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
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

        matrix_dictionary: Dict[str, sps.spmatrix] = data_edge[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        mg: pp.MortarGrid = data_edge["mortar_grid"]

        # We know the number of dofs from the primary and secondary side from their
        # discretizations
        dof = np.array([matrix[0, 0].shape[1], matrix[1, 1].shape[1], mg.num_cells])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # Trace operator for higher-dimensional grid
        trace_primary: sps.spmatrix = matrix_dictionary[self.trace_primary_matrix_key]
        # Associate faces on the higher-dimensional grid with cells
        inv_trace_primary: sps.spmatrix = matrix_dictionary[
            self.inv_trace_primary_matrix_key
        ]

        # Upwind operators
        upwind_primary: sps.spmatrix = matrix_dictionary[self.upwind_primary_matrix_key]
        upwind_secondary: sps.spmatrix = matrix_dictionary[
            self.upwind_secondary_matrix_key
        ]
        flux: sps.spmatrix = matrix_dictionary[self.flux_matrix_key]

        # The mortar variable itself.
        mortar_discr: sps.spmatrix = matrix_dictionary[self.mortar_discr_matrix_key]

        # assemble matrices
        # Note the sign convention: The Darcy mortar flux is positive if it goes
        # from g_h to g_l. Thus a positive transport flux (assuming positive
        # concentration) will go out of g_h, into g_l.

        # Transport out of upper equals lambda.
        # Use integrated projcetion operator; the flux is an extensive quantity
        cc[0, 2] = inv_trace_primary * mg.mortar_to_primary_int()

        # transport out of lower is -lambda
        cc[1, 2] = -mg.mortar_to_secondary_int()

        # Discretisation of mortars
        # If fluid flux(lam_flux) is positive we use the upper value as weight,
        # i.e., T_primaryat * fluid_flux = lambda.
        # We set cc[2, 0] = T_primaryat * fluid_flux
        # Use averaged projection operator for an intensive quantity
        cc[2, 0] = flux * upwind_primary * mg.primary_to_mortar_avg() * trace_primary

        # If fluid flux is negative we use the lower value as weight,
        # i.e., T_check * fluid_flux = lambda.
        # we set cc[2, 1] = T_check * fluid_flux
        # Use averaged projection operator for an intensive quantity
        cc[2, 1] = flux * upwind_secondary * mg.secondary_to_mortar_avg()

        # The rhs of T * fluid_flux = lambda
        # Recover the information for the grid-grid mapping
        cc[2, 2] = -mortar_discr

        if data_primary["node_number"] == data_secondary["node_number"]:
            # All contributions to be returned to the same block of the
            # global matrix in this case
            cc = np.array([np.sum(cc, axis=(0, 1))])

        # rhs is zero
        rhs = np.array(
            [np.zeros(dof[0]), np.zeros(dof[1]), np.zeros(dof[2])], dtype=object
        )
        if rhs.ndim == 2:
            # Special case if all elements in dof are 1, numpy interprets the
            # definition of rhs a bit special then.
            rhs = rhs.ravel()

        matrix += cc
        return matrix, rhs

    @pp.time_logger(sections=module_sections)
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
