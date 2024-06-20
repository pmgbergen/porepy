from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization, InterfaceDiscretization
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


class Upwind(Discretization):
    """Discretize a hyperbolic transport equation using a single point upstream
    weighting scheme.

    """

    def __init__(self, keyword: str = "transport") -> None:
        self.keyword = keyword

        # Keywords used to store matrix and right-hand side in the matrix_dictionary
        self.upwind_matrix_key = "transport"
        self.bound_transport_dir_matrix_key = "rhs_dir"
        self.bound_transport_neu_matrix_key = "rhs_neu"

        # Key used to set the advective flux in the parameter dictionary
        self._flux_array_key = "darcy_flux"

    def ndof(self, sd: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (concentration dof).

        Parameters:
            sd: grid, or a subclass.

        Returns:
            The number of degrees of freedom.

        """
        return sd.num_cells

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix for an upwind discretization of a linear transport
        problem.

        To stay true with a legacy format, the assembled system includes scaling with
        the advective flux field.

        Parameters:
            sd: Computational grid, with geometry fields computed.
            data: With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.
                Size: sd.num_cells x sd.num_cells.
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.

        """
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]
        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]

        upwind = matrix_dictionary[self.upwind_matrix_key]

        # Scaling with the advective flux.
        # This is included to stay compatible with the legacy contract for this
        # function (e.g. it should assemble the discretization matrix for the full
        # advection problem).
        flux_arr = parameter_dictionary[self._flux_array_key]
        flux_mat = sps.dia_matrix((flux_arr, 0), shape=(sd.num_faces, sd.num_faces))

        div: sps.spmatrix = pp.fvutils.scalar_divergence(sd)

        if div.shape[1] != upwind.shape[0]:
            # It should not be difficult to fix this, however it requires some thinking
            # on data format for boundary conditions for systems of equations.
            raise ValueError(
                """Dimension mismatch in assembly of discretization term.
                                Be aware that upwinding with multiple components is only
                                supported in Ad mode.
                            """
            )
        matrix = div * flux_mat * upwind

        # Assemble right-hand side.
        bc_values: np.ndarray = parameter_dictionary["bc_values"]
        bc_discr_dir: sps.spmatrix = matrix_dictionary[
            self.bound_transport_dir_matrix_key
        ]
        bc_discr_neu: sps.spmatrix = matrix_dictionary[
            self.bound_transport_neu_matrix_key
        ]

        assert bc_discr_dir.shape == bc_discr_neu.shape
        if (
            div.shape[1] != bc_discr_dir.shape[0]
            or bc_discr_dir.shape[1] != bc_values.size
        ):
            # It should not be difficult to fix this, however it requires some thinking
            # on data format for boundary conditions for systems of equations.
            raise ValueError(
                """Dimension mismatch in assembly of rhs term. Be aware that upwinding
                with multiple components is only supported in Ad mode.
                """
            )
        rhs = div * (bc_discr_neu + bc_discr_dir * flux_mat) * bc_values
        return matrix, rhs

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """Return the matrix and righ-hand side for an upstream discretization based on
        a scalar flux field.

        The vector field is assumed to be given as the normal velocity, weighted with
        the face area, at each face. The discretization is *not* scaled with the fluxes,
        this must be done externally.

        If not specified the inflow boundary conditions are no-flow, while
        the outflow boundary conditions are open.

        The name of data in the input dictionary (data) are:
        darcy_flux : array (sd.num_faces)
            Normal velocity at each face, weighted by the face area.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.
        source : array (sd.num_cells) of source (positive) or sink (negative) terms.
        num_components (int, optional): Number of components to be advected. Defaults
            to 1.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            sparse csr (sd.num_cells, g_num_cells)

            Upwind matrix obtained from the discretization.
            array (g_num_cells)

            Right-hand side which contains the boundary conditions.

        """

        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        # Shortcut for point grids
        if sd.dim == 0:
            matrix_dictionary[self.upwind_matrix_key] = sps.csr_matrix((0, 1))
            matrix_dictionary[self.bound_transport_dir_matrix_key] = sps.csr_matrix(
                (0, 0)
            )
            matrix_dictionary[self.bound_transport_neu_matrix_key] = sps.csr_matrix(
                (0, 0)
            )
            return

        # Get the sign of the advective flux
        darcy_flux: np.ndarray = np.sign(parameter_dictionary[self._flux_array_key])

        bc: pp.BoundaryCondition = parameter_dictionary["bc"]

        # Booleans of flux direction
        pos_flux = darcy_flux >= 0
        neg_flux = np.logical_not(pos_flux)

        # Array to store index of the cell in the upstream direction
        upstream_cell_ind = np.zeros(sd.num_faces, dtype=int)
        # Fill the array based on the cell-face relation. By construction, the normal
        # vector of a face points from the first to the second row in this array
        cf_dense = sd.cell_face_as_dense()
        # Positive fluxes point in the same direction as the normal vector, find the
        # upstream cell
        upstream_cell_ind[pos_flux] = cf_dense[0, pos_flux]
        upstream_cell_ind[neg_flux] = cf_dense[1, neg_flux]

        # Make row and data arrays, preparing to make a coo-matrix for the upstream
        # cell-to-face map.
        row = np.arange(sd.num_faces)
        values = np.ones(sd.num_faces, dtype=int)

        # We need to eliminate faces on the boundary; these will be discretized
        # separately below. On faces with Neumann conditions, boundary conditions apply
        # for inflow; outflow faces should be assigned Dirichlet conditions. For
        # Dirichlet, only inflow conditions are given; for outflow, we use upstream
        # weighting (thus no need to modify the matrix we are about to build).

        # faces with Neumann conditions
        neumann_ind = np.where(bc.is_neu)[0]

        # Faces with Dirichlet conditions and inflow. The latter is identified by
        # considering the direction of the flux, and the upstream element in cf_dense
        # (note that the exterior of the domain is represented by -1 in cf_dense).
        inflow_ind = np.where(
            np.logical_and(
                bc.is_dir,
                np.logical_or(
                    np.logical_and(pos_flux, cf_dense[0] < 0),
                    np.logical_and(neg_flux, cf_dense[1] < 0),
                ),
            )
        )[0]

        # Delete indices that should be treated by boundary conditions
        delete_ind = np.sort(np.r_[neumann_ind, inflow_ind])
        row = np.delete(row, delete_ind)
        values = np.delete(values, delete_ind)
        col = np.delete(upstream_cell_ind, delete_ind)

        # Finally, we can construct the upstream weighting matrix.
        upstream_mat = sps.coo_matrix(
            (
                values,
                (row, col),
            ),
            shape=(sd.num_faces, sd.num_cells),
        ).tocsr()

        # Form and store discretization matrix
        # Expand the discretization matrix to more than one component
        num_components: int = parameter_dictionary.get("num_components", 1)
        matrix_dictionary[self.upwind_matrix_key] = sps.kron(
            upstream_mat, sps.eye(num_components)
        ).tocsr()

        # Boundary conditions
        # Since the upwind discretization could be combined with a diffusion
        # discretization in an advection-diffusion equation, treatment of boundary
        # conditions can be a bit delicate, and the code should be used with some
        # caution. The below implementation follows the following steps:
        #
        # 1) On Neumann boundaries the prescribed boundary value should effectively be
        # added to the adjacent cell, with the convention that influx (so negative
        # boundary value) should correspond to accumulation.
        # 2) On Dirichlet boundaries, we consider only inflow boundaries. Outflow
        # boundaries are treated by the standard discretization.

        # For Neumann faces we need to assign the sign of the divergence, to counteract
        # multiplication with the same sign when the divergence is applied (e.g. in
        # self.assemble_matrix).
        sgn_div = pp.fvutils.scalar_divergence(sd).sum(axis=0).A.squeeze()

        # Need minus signs on both Neumann and Dirichlet data to ensure that
        # accumulation follows from negative fluxes.
        bc_discr_neu = sps.coo_matrix(
            (sgn_div[neumann_ind], (neumann_ind, neumann_ind)),
            shape=(sd.num_faces, sd.num_faces),
        ).tocsr()
        bc_discr_dir = sps.coo_matrix(
            (np.ones(inflow_ind.size), (inflow_ind, inflow_ind)),
            shape=(sd.num_faces, sd.num_faces),
        ).tocsr()

        # Expand matrix to the right number of components, and store it
        matrix_dictionary[self.bound_transport_neu_matrix_key] = sps.kron(
            bc_discr_neu, sps.eye(num_components)
        ).tocsr()
        matrix_dictionary[self.bound_transport_dir_matrix_key] = sps.kron(
            bc_discr_dir, sps.eye(num_components)
        ).tocsr()

    def cfl(self, sd: pp.Grid, data: dict, d_name="darcy_flux"):
        """
        Return the time step according to the CFL condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        darcy_flux: array (sd.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters:
            g: grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.
            d_name: keyword for discharge entry in data dictionary

        Returns:
            Time step according to CFL condition.

        """
        if sd.dim == 0:
            return np.inf
        # Retrieve the data
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        darcy_flux = parameter_dictionary[d_name]
        phi = parameter_dictionary["mass_weight"]

        faces, cells, _ = sparse_array_to_row_col_data(sd.cell_faces)

        # Detect and remove the faces which have zero in darcy_flux
        not_zero = ~np.isclose(np.zeros(faces.size), darcy_flux[faces], atol=0)
        if not np.any(not_zero):
            return np.inf

        cells = cells[not_zero]
        faces = faces[not_zero]

        # Compute discrete distance cell to face centers
        dist_vector = sd.face_centers[:, faces] - sd.cell_centers[:, cells]
        # Element-wise scalar products between the distance vectors and the
        # normals
        dist = np.einsum("ij,ij->j", dist_vector, sd.face_normals[:, faces])
        # Additionally, we consider the phi (porosity) and the cell-mapping
        coeff = phi[cells]
        # deltaT is deltaX/darcy_flux with coefficient
        return np.amin(np.abs(np.divide(dist, darcy_flux[faces])) * coeff)

    def darcy_flux(self, sd: pp.Grid, beta, cell_apertures=None):
        """
        Return the normal component of the velocity, for each face, weighted by
        the face area and aperture.

        Parameters:
            g: grid, or a subclass, with geometry fields computed.
            beta: (3x1) array which represents the constant velocity.
            cell_apertures: (sd.num_faces) array of apertures

        Returns:
            array (sd.num_faces)

                Normal velocity at each face, weighted by the face area.

        """
        if cell_apertures is None:
            face_apertures = np.ones(sd.num_faces)
        else:
            face_apertures = abs(sd.cell_faces) * cell_apertures
            r, _, _ = sparse_array_to_row_col_data(sd.cell_faces)
            face_apertures = face_apertures / np.bincount(r)

        beta = np.asarray(beta)
        assert beta.size == 3

        if sd.dim == 0:
            if sd.num_faces == 0:
                dot_prod = np.zeros(0)
            else:
                dot_prod = np.dot(sd.face_normals.ravel("F"), face_apertures * beta)
            return np.atleast_1d(dot_prod)

        return np.array(
            [np.dot(n, a * beta) for n, a in zip(sd.face_normals.T, face_apertures)]
        )


class UpwindCoupling(InterfaceDiscretization):
    def __init__(self, keyword: str) -> None:
        # Keywords for accessing discretization matrices
        self.keyword = keyword

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

        self._flux_array_key = "darcy_flux"

    def key(self) -> str:
        return self.keyword + "_"

    def discretization_key(self):
        return self.key() + pp.DISCRETIZATION

    def ndof(self, intf: pp.MortarGrid) -> int:
        return intf.num_cells

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
    ) -> None:
        # First check if the grid dimensions are compatible with the implementation. It
        # is not difficult to cover the case of equal dimensions, it will require trace
        # operators for both grids, but it has not yet been done.
        if sd_primary.dim - sd_secondary.dim not in [1, 2]:
            raise ValueError(
                "Implementation is only valid for grids one dimension apart."
            )

        matrix_dictionary = data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]

        # Normal component of the velocity from the higher dimensional grid
        lam_flux: np.ndarray = np.sign(
            data_intf[pp.PARAMETERS][self.keyword][self._flux_array_key]
        )

        # mapping from upper dim cells to faces
        # The mortars always points from upper to lower, so we don't flip any signs. The
        # mapping will be non-zero also for faces not adjacent to the mortar grid,
        # however, we wil hit it with mortar projections, thus kill those elements
        inv_trace_h = np.abs(pp.fvutils.scalar_divergence(sd_primary))
        # We also need a trace-like projection from cells to faces
        trace_h = inv_trace_h.T

        matrix_dictionary[self.inv_trace_primary_matrix_key] = inv_trace_h
        matrix_dictionary[self.trace_primary_matrix_key] = trace_h

        # Find upwind weighting. if flag is True we use the upper weights if flag is
        # False we use the lower weighs
        flag = (lam_flux > 0).astype(float)
        not_flag = 1 - flag

        # Discretizations are the flux, but masked so that only the upstream direction
        # is hit.
        upwind_from_primary = sps.diags(flag)
        upwind_from_secondary = sps.diags(not_flag)

        flux = sps.diags(lam_flux)

        matrix_dictionary[self.upwind_primary_matrix_key] = upwind_from_primary
        matrix_dictionary[self.upwind_secondary_matrix_key] = upwind_from_secondary
        matrix_dictionary[self.flux_matrix_key] = flux

        # Identity matrix, to represent the mortar variable itself
        matrix_dictionary[self.mortar_discr_matrix_key] = sps.eye(intf.num_cells)

    def assemble_matrix_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf,
        matrix: sps.spmatrix,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """
        Construct the matrix (and right-hand side) for the coupling conditions.
        Note: the right-hand side is not implemented now.

        Parameters:
            sd_primary: grid of higher dimension
            sd_secondary: grid of lower dimension
            data_primary: dictionary which stores the data for the higher dimensional
                grid
            data_secondary: dictionary which stores the data for the lower dimensional
                grid
            data_intf: dictionary which stores the data for the edges of the mdg
            matrix: Uncoupled discretization matrix.

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.

        """

        matrix_dictionary: Dict[str, sps.spmatrix] = data_intf[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions

        # We know the number of dofs from the primary and secondary side from their
        # discretizations
        dof = np.array([matrix[0, 0].shape[1], matrix[1, 1].shape[1], intf.num_cells])
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

        # The advective flux
        lam_flux: np.ndarray = np.abs(
            data_intf[pp.PARAMETERS][self.keyword][self._flux_array_key]
        )
        scaling = sps.dia_matrix((lam_flux, 0), shape=(intf.num_cells, intf.num_cells))

        # assemble matrices

        # Note the sign convention: The Darcy mortar flux is positive if it goes from
        # sd_primary to sd_secondary. Thus, a positive transport flux (assuming positive
        # concentration) will go out of sd_primary, into sd_secondary.

        # Transport out of upper equals lambda.
        # Use integrated projection operator; the flux is an extensive quantity
        cc[0, 2] = inv_trace_primary * intf.mortar_to_primary_int()

        # transport out of lower is -lambda
        cc[1, 2] = -intf.mortar_to_secondary_int()

        # Discretisation of mortars
        # If fluid flux(lam_flux) is positive we use the upper value as weight,
        # i.e., T_primaryat * fluid_flux = lambda.
        # We set cc[2, 0] = T_primaryat * fluid_flux
        # Use averaged projection operator for an intensive quantity
        cc[2, 0] = (
            scaling
            * flux
            * upwind_primary
            * intf.primary_to_mortar_avg()
            * trace_primary
        )

        # If fluid flux is negative we use the lower value as weight,
        # i.e., T_check * fluid_flux = lambda.
        # we set cc[2, 1] = T_check * fluid_flux
        # Use averaged projection operator for an intensive quantity
        cc[2, 1] = scaling * flux * upwind_secondary * intf.secondary_to_mortar_avg()

        # The rhs of T * fluid_flux = lambda
        # Recover the information for the grid-grid mapping
        cc[2, 2] = -mortar_discr

        if sd_primary == sd_secondary:
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
