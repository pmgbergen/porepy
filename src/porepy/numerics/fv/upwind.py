from __future__ import annotations

from typing import Any, Optional

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

        # Keywords used to store matrix and right-hand side in the matrix_dictionary.
        self.upwind_matrix_key = "transport"
        """Keyword used to identify the discretization matrix for contributions from
        the domain. Defaults to 'transport'.

        """
        self.bound_transport_dir_matrix_key = "rhs_dir"
        """Keyword used to identify the discretization matrix for contributions from
        Dirichlet boundaries. Defaults to 'rhs_dir'.

        """
        self.bound_transport_neu_matrix_key = "rhs_neu"
        """Keyword used to identify the discretization matrix for contributions from
        Neumann boundaries. Defaults to 'rhs_neu'.

        """

        # Key used to set the advective flux in the parameter dictionary.
        self._flux_array_key = "darcy_flux"
        """Keyword used to identify the parameter matrix for face fluxes. Defaults to
        'darcy_flux'.

        """

    @property
    def flux_array_key(self) -> str:
        return self._flux_array_key

    @flux_array_key.setter
    def flux_array_key(self, value: str) -> None:
        self._flux_array_key = value

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method. In this
        case number of cells.

        Parameters:
            sd: Subdomain grid.

        Returns:
            The number of degrees of freedom.

        """
        return sd.num_cells

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side for an upwind discretization of a
        linear transport problem.

        To stay true with a legacy format, the assembled system includes scaling with
        the advective flux field.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            - parameter_dictionary, storing all parameters. Stored in
              ``data[pp.PARAMETERS][self.keyword]``.
            - matrix_dictionary, for storage of discretization matrices. Stored in
              ``data[pp.DISCRETIZATION_MATRICES][self.keyword]``.

        parameter_dictionary contains the entries:
            - bc_values: :class:`~numpy.ndarray` of
              ``shape=(boundary_grid.num_cells,)``.

        matrix_dictionary contains the entries:
            - ``self.upwind_matrix_key``: :class:`~scipy.sparse.csr_matrix` of
                ``shape=(sd.num_faces, sd.num_cells)``. Upwind matrix obtained from the
                discretization.
            - ``self.bound_transport_dir_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(sd.num_faces, sd.num_faces)``. Right-hand side containing
                the discretization matrix for contributions from Dirichlet boundary
                conditions.
            - ``self.bound_transport_neu_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(sd.num_faces, sd.num_faces)``. Right-hand side containing
                the discretization matrix for contributions from Neumann boundary
                conditions.

        The matrix_dictionary entries are normally set by calling
        :meth:`Upwind.discretize`.

        Parameters:
            sd: Computational grid, with geometry fields computed.
            data: Dictionary containing stored discretization data and parameters.

        Returns:
            scipy.sparse.csr_matrix: ``shape=(sd.num_cells, sd.num_cells)`` System
                matrix of this discretization.
            np.ndarray: ``shape=(sd.num_cells,)`` Right hand side vector with
                representation of boundary conditions.

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

        div: sps.spmatrix = sd.divergence(dim=1)

        if div.shape[1] != upwind.shape[0]:
            # It should not be difficult to fix this, however it requires some thinking
            # on data format for boundary conditions for systems of equations.
            raise ValueError(
                """Dimension mismatch in assembly of discretization term.
                                Be aware that upwinding with multiple components is only
                                supported in Ad mode.
                """
            )
        matrix = div @ flux_mat @ upwind

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
        rhs = div @ (bc_discr_neu + bc_discr_dir @ flux_mat) @ bc_values
        return matrix, rhs

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """Discretize the matrix and right-hand side for an upstream discretization
        based on a scalar flux field.

        The vector field is assumed to be given as the normal velocity, weighted with
        the face area, at each face. The discretization is *not* scaled with the fluxes,
        this must be done externally.

        If not specified the inflow boundary conditions are no-flow, while
        the outflow boundary conditions are open.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            - parameter_dictionary, storing all parameters. Stored in
              ``data[pp.PARAMETERS][self.keyword]``.
            - matrix_dictionary, for storage of discretization matrices. Stored in
              ``data[pp.DISCRETIZATION_MATRICES][self.keyword]``.

        parameter_dictionary contains the entries:
            - bc: :class:`~porepy.params.BoundaryCondition`.
                Boundary conditions for the advected property.
            - ``self._flux_array_key``: :class:`~numpy.ndarray` of
              ``shape=(sd.num_faces,)``. Normal velocity at each face, weighted by the
              face area.
            - num_components: ``int`` (optional). Number of components to be advected.
              Defaults to 1.

        matrix_dictionary will be updated with the following entries:
            - ``self.upwind_matrix_key``: :class:`~scipy.sparse.csr_matrix` of
                ``shape=(sd.num_faces, sd.num_cells)``. Upwind matrix obtained from the
                discretization.
            - ``self.bound_transport_dir_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(sd.num_faces, sd.num_faces)``. Right-hand side containing
                the discretization matrix for contributions from Dirichlet boundary
                conditions.
            - ``self.bound_transport_neu_matrix_key````:
              :class:`~scipy.sparse.csr_matrix` of
              ``shape=(sd.num_faces, sd.num_faces)``. Right-hand side containing the
              discretization matrix for contributions from Neumann boundary conditions.

        Parameters:
            sd: Grid, or a subclass, with geometry fields computed.
            data: Dictionary to store the data.


        """
        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        # Shortcut for point grids.
        if sd.dim == 0:
            matrix_dictionary[self.upwind_matrix_key] = sps.csr_matrix((0, 1))
            matrix_dictionary[self.bound_transport_dir_matrix_key] = sps.csr_matrix(
                (0, 0)
            )
            matrix_dictionary[self.bound_transport_neu_matrix_key] = sps.csr_matrix(
                (0, 0)
            )
            return

        # Get the sign of the advective flux.
        darcy_flux: np.ndarray = np.sign(parameter_dictionary[self._flux_array_key])

        # Enables the creation of an upwind object even if boundary data is not
        # externally provided.
        if "bc" in parameter_dictionary:
            bc: pp.BoundaryCondition = parameter_dictionary["bc"]
        else:
            # Set a Dirichlet condition by default. Motivation (from Omar Duran): If the
            # advecting flux is non-zero on external facets, this choice ensures
            # consistent handling of sinking phases.
            bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")

        # Booleans of flux direction.
        pos_flux = darcy_flux >= 0
        neg_flux = np.logical_not(pos_flux)

        # Array to store index of the cell in the upstream direction.
        upstream_cell_ind = np.zeros(sd.num_faces, dtype=int)
        # Fill the array based on the cell-face relation. By construction, the normal
        # vector of a face points from the first to the second row in this array
        cf_dense = sd.cell_faces_as_dense()
        # Positive fluxes point in the same direction as the normal vector, find the
        # upstream cell.
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

        # Faces with Neumann conditions.
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

        # Delete indices that should be treated by boundary conditions.
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

        # Form and store discretization matrix.
        # Expand the discretization matrix to more than one component.
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
        sgn_div = np.asarray(sd.divergence(dim=1).sum(axis=0)).squeeze()

        bc_discr_neu = sps.coo_matrix(
            (sgn_div[neumann_ind], (neumann_ind, neumann_ind)),
            shape=(sd.num_faces, sd.num_faces),
        ).tocsr()
        bc_discr_dir = sps.coo_matrix(
            (np.ones(inflow_ind.size), (inflow_ind, inflow_ind)),
            shape=(sd.num_faces, sd.num_faces),
        ).tocsr()

        # Expand matrix to the right number of components, and store it.
        matrix_dictionary[self.bound_transport_neu_matrix_key] = sps.kron(
            bc_discr_neu, sps.eye(num_components)
        ).tocsr()
        matrix_dictionary[self.bound_transport_dir_matrix_key] = sps.kron(
            bc_discr_dir, sps.eye(num_components)
        ).tocsr()

    def darcy_flux(
        self, sd: pp.Grid, beta: np.ndarray, cell_apertures=None
    ) -> np.ndarray:
        """Return the normal component of the velocity, for each face, weighted by the
        face area and aperture.

        Parameters:
            sd: Grid, or a subclass, with geometry fields computed.
            beta: ``shape=(3,1)``
                Array which represents the constant velocity.
            cell_apertures: ``shape=(sd.num_cells,)``
                Array of apertures

        Returns:
            array: ``shape=(sd.num_faces)``
                Normal velocity at each face, weighted by the face area.

        """
        if cell_apertures is None:
            face_apertures = np.ones(sd.num_faces)
        else:
            face_apertures = abs(sd.cell_faces) @ cell_apertures
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
        self.keyword = keyword
        """Keyword for accessing discretization matrices in the matrix_dictionary.
        Defaults to 'trace'."""
        self.trace_primary_matrix_key = "trace"
        """Keyword used to identify the trace operator for the primary grid.
        Defaults to 'trace'."""

        self.inv_trace_primary_matrix_key = "inv_trace"
        """Keyword used to identify the inverse trace operator (face -> cell).
        Defaults to 'inv_trace'."""

        self.upwind_primary_matrix_key = "upwind_primary"
        """Keyword used to identify the matrix for filtering upwind values from the
        primary grid. Defaults to 'upwind_primary'."""

        self.upwind_secondary_matrix_key = "upwind_secondary"
        """Keyword used to identify the matrix for filtering upwind values from the
        secondary grid. Defaults to 'upwind_secondary'."""

        self.flux_matrix_key = "flux"
        """Keyword used to identify the matrix that carries the fluxes.
        Defaults to 'flux'."""

        self.mortar_discr_matrix_key = "mortar_discr"
        """Keyword used to identify the discretization of the mortar variable.
        Defaults to 'mortar_discr'."""

        self._flux_array_key = "darcy_flux"
        """Keyword used to identify the parameter matrix for face fluxes.
        Defaults to 'darcy_flux'."""

    def key(self) -> str:
        return self.keyword + "_"

    def discretization_key(self):
        return self.key() + pp.DISCRETIZATION

    @property
    def flux_array_key(self) -> str:
        return self._flux_array_key

    @flux_array_key.setter
    def flux_array_key(self, value: str) -> None:
        self._flux_array_key = value

    def ndof(self, intf: pp.MortarGrid) -> int:
        return intf.num_cells

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: dict,
        data_secondary: dict,
        data_intf: dict,
    ) -> None:
        """Discretize the matrix for a coupled upstream discretization based on a scalar
        flux field on the higher dimensional domain.

        In the following, we denote with "primary" the higher-dimensional domain and
        with "secondary" the lower-dimensional domain.

        We assume the following two sub-dictionaries to be present in the ``data_intf``
        dictionary:
            - parameter_dictionary, storing all parameters. Stored in
              ``data_intf[pp.PARAMETERS][self.keyword]``.
            - matrix_dictionary, for storage of discretization matrices. Stored in
              ``data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]``.

        parameter_dictionary contains the entries:
            - ``self._flux_array_key``: :class:`~numpy.ndarray` of
              ``shape=(sd_primary.num_faces,)``. Normal velocity at each face of the
              primary grid, weighted by the face area.

        matrix_dictionary will be updated with the following entries:
            - ``self.inv_trace_primary_matrix_key``: :class:`~scipy.sparse.csr_matrix`
              of ``shape=(sd_primary.num_faces, sd_primary.num_cells)``. Inverse trace
              operator.
            - ``self.trace_primary_matrix_key``: :class:`~scipy.sparse.csr_matrix` of
              ``shape=(sd_primary.num_faces, sd_primary.num_cells)``. Trace operator.
            - ``self.upwind_primary_matrix_key``: :class:`~scipy.sparse.csr_matrix` of
                ``shape=(sd.num_faces, sd.num_cells)``. Upwind matrix for the
                primary domain.
            - ``self.upwind_secondary_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(sd.num_faces, sd.num_faces)``. Upwind matrix for the
                secondary domain.
            - ``self.flux_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(sd.num_faces, sd.num_faces)``. Flux matrix.
            - ``self.mortar_discr_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(intf.num_cells, intf.num_cells)``. Identity matrix for the
                mortar variable.

        Parameters:
            sd_primary: Grid, or a subclass, of the primary domain, with geometry fields
                computed.
            sd_secondary: Grid, or a subclass, of the secondary domain, with geometry
                fields computed.
            intf: MortarGrid, or a subclass, of the interface domain, with geometry
                fields computed.
            data_primary: Data dictionary for the primary domain.
            data_secondary: Data dictionary for the secondary domain.
            data_intf: Data dictionary for the interface domain.

        """

        # First check if the grid dimensions are compatible with the implementation. It
        # is not difficult to cover the case of equal dimensions, it will require trace
        # operators for both grids, but it has not yet been done.
        if sd_primary.dim - sd_secondary.dim not in [1, 2]:
            raise ValueError(
                "Implementation is only valid for grids one dimension apart."
            )

        matrix_dictionary = data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]

        # Normal component of the velocity from the higher dimensional grid.
        lam_flux: np.ndarray = np.sign(
            data_intf[pp.PARAMETERS][self.keyword][self._flux_array_key]
        )

        # Mapping from upper dim cells to faces.
        # The mortars always points from upper to lower, so we don't flip any signs. The
        # mapping will be non-zero also for faces not adjacent to the mortar grid,
        # however, we wil hit it with mortar projections, thus kill those elements.
        inv_trace_h = np.abs(sd_primary.divergence(dim=1))
        # We also need a trace-like projection from cells to faces.
        trace_h = inv_trace_h.T

        matrix_dictionary[self.inv_trace_primary_matrix_key] = inv_trace_h
        matrix_dictionary[self.trace_primary_matrix_key] = trace_h

        # Find upwind weighting. if flag is True we use the upper weights if flag is
        # False we use the lower weights.
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

        # Identity matrix, to represent the mortar variable itself.
        matrix_dictionary[self.mortar_discr_matrix_key] = sps.eye(intf.num_cells)

    def assemble_matrix_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: dict,
        data_secondary: dict,
        data_intf: dict,
        matrix: sps.spmatrix,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Construct the matrix (and right-hand side) for the coupling conditions.

        Note:
            The right-hand side is not implemented now.

        In the following, we denote with "primary" the higher-dimensional domain and
        with "secondary" the lower-dimensional domain.

        We assume the following two sub-dictionaries to be present in the ``data_intf``
        dictionary:
            - parameter_dictionary, storing all parameters. Stored in
              ``data_intf[pp.PARAMETERS][self.keyword]``.
            - matrix_dictionary, for storage of discretization matrices. Stored in
              ``data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]``.

        parameter_dictionary contains the entries:
            - ``self._flux_array_key``: :class:`~numpy.ndarray` of
              ``shape=(sd_primary.num_faces,)``. Normal velocity at each face of the
              primary grid, weighted by the face area.

        matrix_dictionary contains the following entries:
            - ``self.inv_trace_primary_matrix_key``: :class:`~scipy.sparse.csr_matrix`
              of ``shape=(sd_primary.num_faces, sd_primary.num_cells)``. Inverse trace
              operator.
            - ``self.trace_primary_matrix_key``: :class:`~scipy.sparse.csr_matrix` of
              ``shape=(sd_primary.num_faces, sd_primary.num_cells)``. Trace operator.
            - ``self.upwind_primary_matrix_key``: :class:`~scipy.sparse.csr_matrix` of
                ``shape=(sd.num_faces, sd.num_cells)``. Upwind matrix for the
                primary domain.
            - ``self.upwind_secondary_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(sd.num_faces, sd.num_faces)``. Upwind matrix for the
                secondary domain.
            - ``self.flux_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(sd.num_faces, sd.num_faces)``. Flux matrix.
            - ``self.mortar_discr_matrix_key``: :class:`~scipy.sparse.csr_matrix`
                of ``shape=(intf.num_cells, intf.num_cells)``. Identity matrix for the
                mortar variable.

        Parameters:
            sd_primary: Grid of the primary domain.
            sd_secondary: Grid of the secondary domain.
            intf: MortarGrid of the interface domain.
            data_primary: Data dictionary for the primary domain.
            data_secondary: Data dictionary for the secondary domain.
            data_intf: Data dictionary for the edges of the mixed-dimensional grid.
            matrix: Uncoupled discretization matrix.

        Returns:
            matrix: Block matrix storing the contribution of the coupling condition. See
            the abstract coupling class for a more detailed description.
            rhs: Right-hand side of the coupling condition. Not implemented.

        """

        matrix_dictionary: dict[str, sps.spmatrix] = data_intf[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        # Retrieve the number of degrees of both grids.
        # Create the block matrix for the contributions.

        # We know the number of dofs from the primary and secondary side from their
        # discretizations.
        dof = np.array([matrix[0, 0].shape[1], matrix[1, 1].shape[1], intf.num_cells])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # Trace operator for higher-dimensional grid.
        trace_primary: sps.spmatrix = matrix_dictionary[self.trace_primary_matrix_key]
        # Associate faces on the higher-dimensional grid with cells.
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

        # Assemble matrices.

        # Note the sign convention: The Darcy mortar flux is positive if it goes from
        # sd_primary to sd_secondary. Thus, a positive transport flux (assuming positive
        # concentration) will go out of sd_primary, into sd_secondary.

        # Transport out of upper equals lambda.
        # Use integrated projection operator; the flux is an extensive quantity.
        cc[0, 2] = inv_trace_primary @ intf.mortar_to_primary_int()

        # Transport out of lower is -lambda.
        cc[1, 2] = -intf.mortar_to_secondary_int()

        # Discretisation of mortars.
        # If fluid flux(lam_flux) is positive we use the upper value as weight,
        # i.e., T_primaryat @ fluid_flux = lambda.
        # We set cc[2, 0] = T_primaryat @ fluid_flux.
        # Use averaged projection operator for an intensive quantity.
        cc[2, 0] = (
            scaling
            @ flux
            @ upwind_primary
            @ intf.primary_to_mortar_avg()
            @ trace_primary
        )

        # If fluid flux is negative we use the lower value as weight,
        # i.e., T_check @ fluid_flux = lambda.
        # We set cc[2, 1] = T_check @ fluid_flux.
        # Use averaged projection operator for an intensive quantity.
        cc[2, 1] = scaling @ flux @ upwind_secondary @ intf.secondary_to_mortar_avg()

        # The rhs of T @ fluid_flux = lambda.
        # Recover the information for the grid-grid mapping.
        cc[2, 2] = -mortar_discr

        if sd_primary == sd_secondary:
            # All contributions to be returned to the same block of the
            # global matrix in this case.
            cc = np.array([np.sum(cc, axis=(0, 1))])

        # rhs is zero.
        rhs = np.array(
            [np.zeros(dof[0]), np.zeros(dof[1]), np.zeros(dof[2])], dtype=object
        )
        if rhs.ndim == 2:
            # Special case if all elements in dof are 1, numpy interprets the
            # definition of rhs a bit special then.
            rhs = rhs.ravel()

        matrix += cc
        return matrix, rhs


def _dirichlet_inflow(
    bc: pp.BoundaryCondition,
    pos_flux: np.ndarray,
    neg_flux: np.ndarray,
    cf_dense: np.ndarray,
) -> np.ndarray:
    """Dirichlet boundary faces that are INFLOW for the given flux direction (the upstream cell is
    the exterior). These are dropped from the transport matrix -- handled by the boundary term."""
    return np.where(
        np.logical_and(
            bc.is_dir,
            np.logical_or(
                np.logical_and(pos_flux, cf_dense[0] < 0),
                np.logical_and(neg_flux, cf_dense[1] < 0),
            ),
        )
    )[0]


def _single_point_upwind_matrices(
    sd: pp.Grid,
    flux_array: np.ndarray,
    bc: pp.BoundaryCondition,
    num_components: int,
    cache: Optional[dict] = None,
) -> tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]:
    """Single-point upstream-weighting matrices for one signed flux array.

    Faithful extraction of :meth:`Upwind.discretize` so it can be reused for *two*
    directions in one discretization (see :class:`HUpwind`). Returns
    ``(upwind, bound_transport_dir, bound_transport_neu)``.

    DATA-ONLY FAST PATH. The upwind sparsity is FIXED (fixed-sparsity weighting below + fixed
    geometry + fixed Neumann set), so re-discretization only changes the per-face *data* (which
    neighbour is upstream). If a caller-owned ``cache`` dict is supplied, the first call builds and
    stores the CSR structure plus the ``values -> data`` scatter map; subsequent calls with an
    UNCHANGED drop-mask skip the COO / ``tocsr`` / sort entirely and just scatter the new data into
    the cached structure. When a Dirichlet face flips inflow/outflow (the only way the mask can
    change) the structure is rebuilt and re-cached. ``cache=None`` (default) always does the full
    build, preserving the original behaviour for other callers. The fast path is bit-identical to
    the full build; it is enabled only for the common ``num_components == 1`` case.
    """
    if sd.dim == 0:
        return (
            sps.csr_matrix((0, num_components)),
            sps.csr_matrix((0, 0)),
            sps.csr_matrix((0, 0)),
        )
    sign_flux = np.sign(flux_array)
    pos_flux = sign_flux >= 0
    neg_flux = np.logical_not(pos_flux)

    # ---- data-only fast path (cached structure, single component, unchanged drop-mask) ----
    st = cache.get("struct") if (cache is not None and num_components == 1) else None
    if st is not None:
        if not st["has_dir"]:
            bc_dir = st["bc_dir"]  # no Dirichlet faces -> the drop-mask is fixed
            fast = True
        else:
            inflow = _dirichlet_inflow(bc, pos_flux, neg_flux, st["cf_dense"])
            drop = st["drop_neu"].copy()
            drop[inflow] = True
            fast = bool(np.array_equal(st["col_ok"] & ~drop[st["row"]], st["keep"]))
            if fast:  # inflow set unchanged too -> refresh only the (small) boundary matrix
                bc_dir = sps.coo_matrix(
                    (np.ones(inflow.size), (inflow, inflow)),
                    shape=(sd.num_faces, sd.num_faces),
                ).tocsr()
        if fast:
            values = np.concatenate([pos_flux.astype(float), neg_flux.astype(float)])
            upwind = sps.csr_matrix(
                (values[st["data_src"]], st["indices"], st["indptr"]),
                shape=st["shape"],
                copy=False,
            )
            upwind.has_sorted_indices = True  # cached indices are sorted -> no re-sort/mutation
            return upwind, bc_dir, st["bc_neu"]

    # ---- full build (first call, changed structure, no cache, or num_components != 1) ----
    cf_dense = sd.cell_faces_as_dense()

    neumann_ind = np.where(bc.is_neu)[0]
    inflow_ind = _dirichlet_inflow(bc, pos_flux, neg_flux, cf_dense)
    drop_face = np.zeros(sd.num_faces, dtype=bool)
    drop_face[np.r_[neumann_ind, inflow_ind]] = True

    # FIXED-SPARSITY single-point upwinding: every face keeps a STRUCTURAL entry for BOTH of
    # its neighbour cells, carrying weight 1 on the upstream cell and an explicit 0 on the
    # downstream cell. The pattern is then purely geometric and does NOT change when the flow
    # direction flips (only the data swaps), while ``upwind @ x`` is bit-identical to the
    # classic one-entry-per-face form (the explicit zero contributes nothing). This lets a
    # compiled assembler bake the Jacobian structure once instead of recompiling per iterate.
    faces = np.arange(sd.num_faces)
    row = np.concatenate([faces, faces])
    col = np.concatenate([cf_dense[0], cf_dense[1]])
    values = np.concatenate([pos_flux.astype(float), neg_flux.astype(float)])
    col_ok = col >= 0
    keep = col_ok & ~drop_face[row]  # drop exterior "cells" and BC-handled faces
    upstream_mat = sps.coo_matrix(
        (values[keep], (row[keep], col[keep])), shape=(sd.num_faces, sd.num_cells)
    ).tocsr()
    upstream_mat.sort_indices()  # canonical, stable pattern

    sgn_div = np.asarray(sd.divergence(dim=1).sum(axis=0)).squeeze()
    bc_discr_neu = sps.coo_matrix(
        (sgn_div[neumann_ind], (neumann_ind, neumann_ind)),
        shape=(sd.num_faces, sd.num_faces),
    ).tocsr()
    bc_discr_dir = sps.coo_matrix(
        (np.ones(inflow_ind.size), (inflow_ind, inflow_ind)),
        shape=(sd.num_faces, sd.num_faces),
    ).tocsr()

    # ``kron(M, eye(1))`` is a no-op; skip it in the (common) single-component case.
    if num_components == 1:
        if cache is not None:
            # Capture the COO -> sorted-CSR scatter: ``upstream_mat.data == values[data_src]``.
            # Each (face, cell) pair is unique, so tocsr is a pure permutation (no summation),
            # and building the same COO with the value-indices as data recovers that permutation.
            keep_idx = np.nonzero(keep)[0]
            perm = sps.coo_matrix(
                (keep_idx.astype(float), (row[keep], col[keep])),
                shape=(sd.num_faces, sd.num_cells),
            ).tocsr()
            perm.sort_indices()
            drop_neu = np.zeros(sd.num_faces, dtype=bool)
            drop_neu[neumann_ind] = True
            cache["struct"] = {
                "indptr": upstream_mat.indptr,
                "indices": upstream_mat.indices,
                "shape": upstream_mat.shape,
                "data_src": perm.data.astype(np.intp),
                "keep": keep,
                "has_dir": bool(np.any(bc.is_dir)),
                "col_ok": col_ok,
                "row": row,
                "drop_neu": drop_neu,
                "cf_dense": cf_dense,
                "bc_dir": bc_discr_dir,
                "bc_neu": bc_discr_neu,
            }
        return upstream_mat, bc_discr_dir, bc_discr_neu
    upwind = sps.kron(upstream_mat, sps.eye(num_components)).tocsr()
    rhs_neu = sps.kron(bc_discr_neu, sps.eye(num_components)).tocsr()
    rhs_dir = sps.kron(bc_discr_dir, sps.eye(num_components)).tocsr()
    return upwind, rhs_dir, rhs_neu


class HUpwind(Upwind):
    """Two-direction upwinding for the simplicial buoyancy term.

    Stores **two** direction arrays and, in :meth:`discretize`, builds one single-point
    upwind matrix (plus its boundary matrices) per direction:

    - ``upwind_gamma`` / ``bound_transport_{dir,neu}_gamma`` -- upstream by ``gamma_flux``;
    - ``upwind_delta`` / ``bound_transport_{dir,neu}_delta`` -- upstream by ``delta_flux``.

    The model sets the two directions per scheme (see
    :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.update_buoyancy_driven_fluxes`):
    hybrid upwinding (HU) stores the inter-phase gravity flux with opposite signs
    (``+ddf(rho_gamma - rho_delta)`` / ``-ddf(...)``); phase-potential upwinding (PPU)
    stores each phase's own potential flux (``Psi_gamma`` / ``Psi_delta``). The matrix
    keys are exposed as AD methods by :func:`~porepy.numerics.ad.ad_utils.wrap_discretization`
    (see :class:`~porepy.numerics.ad.discretizations.HUpwindAd`).
    """

    def __init__(self, keyword: str = "hybrid_upwind") -> None:
        super().__init__(keyword)
        # gamma reuses the base Upwind keys; delta gets its own.
        self.upwind_matrix_key = "transport_gamma"
        self.bound_transport_dir_matrix_key = "rhs_dir_gamma"
        self.bound_transport_neu_matrix_key = "rhs_neu_gamma"
        self.upwind_gamma_matrix_key = "transport_gamma"
        self.bound_transport_dir_gamma_matrix_key = "rhs_dir_gamma"
        self.bound_transport_neu_gamma_matrix_key = "rhs_neu_gamma"
        self.upwind_delta_matrix_key = "transport_delta"
        self.bound_transport_dir_delta_matrix_key = "rhs_dir_delta"
        self.bound_transport_neu_delta_matrix_key = "rhs_neu_delta"
        self._gamma_flux_key = "hybrid_gamma_flux"
        self._delta_flux_key = "hybrid_delta_flux"

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        num_components: int = parameter_dictionary.get("num_components", 1)

        if "bc" in parameter_dictionary:
            bc = parameter_dictionary["bc"]
        else:
            bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")

        # gamma upstream by gamma_flux, delta by delta_flux (two independent directions).
        gamma_dir = np.asarray(parameter_dictionary[self._gamma_flux_key])
        delta_dir = np.asarray(parameter_dictionary[self._delta_flux_key])
        # Per-direction data-only caches, persisted in the (per-grid) ``data`` dict so they are
        # reused across re-discretizations. Keyed by this discretization's keyword so distinct
        # HUpwind instances sharing a grid do not collide.
        fast = data.setdefault("_hu_upwind_fast_cache", {})
        up_g, dir_g, neu_g = _single_point_upwind_matrices(
            sd, gamma_dir, bc, num_components,
            cache=fast.setdefault(self.keyword + ":gamma", {}),
        )
        up_d, dir_d, neu_d = _single_point_upwind_matrices(
            sd, delta_dir, bc, num_components,
            cache=fast.setdefault(self.keyword + ":delta", {}),
        )
        matrix_dictionary["transport_gamma"] = up_g
        matrix_dictionary["rhs_dir_gamma"] = dir_g
        matrix_dictionary["rhs_neu_gamma"] = neu_g
        matrix_dictionary["transport_delta"] = up_d
        matrix_dictionary["rhs_dir_delta"] = dir_d
        matrix_dictionary["rhs_neu_delta"] = neu_d


class HUpwindCoupling(UpwindCoupling):
    """Interface (mortar) counterpart of :class:`HUpwind`: two directions.

    Builds, per stored direction, the mortar upwind matrices
    ``upwind_{primary,secondary}_gamma`` (from ``gamma_flux``) and
    ``upwind_{primary,secondary}_delta`` (from ``delta_flux``), plus the signed ``flux``,
    in one :meth:`discretize`. The geometric ``trace`` / ``inv_trace`` / ``mortar_discr``
    matrices are built once and shared.
    """

    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)
        # Geometric / shared matrices keep the base keys.
        self.trace_primary_matrix_key = "trace"
        self.inv_trace_primary_matrix_key = "inv_trace"
        self.mortar_discr_matrix_key = "mortar_discr"
        # gamma reuses the base direction-dependent keys; delta gets its own.
        self.upwind_primary_matrix_key = "upwind_primary_gamma"
        self.upwind_secondary_matrix_key = "upwind_secondary_gamma"
        self.flux_matrix_key = "flux_gamma"
        self.upwind_primary_gamma_matrix_key = "upwind_primary_gamma"
        self.upwind_secondary_gamma_matrix_key = "upwind_secondary_gamma"
        self.flux_gamma_matrix_key = "flux_gamma"
        self.upwind_primary_delta_matrix_key = "upwind_primary_delta"
        self.upwind_secondary_delta_matrix_key = "upwind_secondary_delta"
        self.flux_delta_matrix_key = "flux_delta"
        self._gamma_flux_key = "hybrid_gamma_flux"
        self._delta_flux_key = "hybrid_delta_flux"

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: dict,
        data_secondary: dict,
        data_intf: dict,
    ) -> None:
        if sd_primary.dim - sd_secondary.dim not in [1, 2]:
            raise ValueError(
                "Implementation is only valid for grids one dimension apart."
            )
        matrix_dictionary = data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data_intf[pp.PARAMETERS][self.keyword]

        inv_trace_h = np.abs(sd_primary.divergence(dim=1))
        matrix_dictionary["inv_trace"] = inv_trace_h
        matrix_dictionary["trace"] = inv_trace_h.T
        matrix_dictionary["mortar_discr"] = sps.eye(intf.num_cells)

        # gamma rides gamma_flux, delta rides delta_flux (two independent directions).
        for suffix, key in (
            ("gamma", self._gamma_flux_key),
            ("delta", self._delta_flux_key),
        ):
            lf = np.sign(parameter_dictionary[key])
            flag = (lf > 0).astype(float)
            matrix_dictionary[f"upwind_primary_{suffix}"] = sps.diags(flag)
            matrix_dictionary[f"upwind_secondary_{suffix}"] = sps.diags(1.0 - flag)
            matrix_dictionary[f"flux_{suffix}"] = sps.diags(lf)
