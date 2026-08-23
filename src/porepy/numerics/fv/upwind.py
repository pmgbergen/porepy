from __future__ import annotations

from typing import Any, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization, InterfaceDiscretization
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


def _fixed_diag(vec: np.ndarray) -> sps.csr_matrix:
    """Diagonal CSR keeping explicit zeros.

    Unlike ``sps.diags``, the sparsity pattern does not change when the flow
    direction flips, so downstream structure caches stay valid.

    The diagonal CSR structure (``indices = arange(n)``, ``indptr = arange(n+1)``) is
    known-valid, so the matrix is assembled directly and scipy's per-construction
    validation (``get_index_dtype`` / ``check_format`` / ``prune``) is skipped. This
    is a hot inner-loop constructor in the upwind discretization (called
    ~n_grids x n_equations times per re-discretization).
    """
    n = int(vec.size)
    m = sps.csr_matrix((n, n))
    m.data = np.ascontiguousarray(vec, dtype=float)
    m.indices = np.arange(n, dtype=np.int32)
    m.indptr = np.arange(n + 1, dtype=np.int32)
    m._has_sorted_indices = True
    m._has_canonical_format = True
    return m


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

        num_components: int = parameter_dictionary.get("num_components", 1)

        # Enables the creation of an upwind object even if boundary data is not
        # externally provided.
        if "bc" in parameter_dictionary:
            bc: pp.BoundaryCondition = parameter_dictionary["bc"]
        else:
            # Set a Dirichlet condition by default. Motivation (from Omar Duran): If the
            # advecting flux is non-zero on external facets, this choice ensures
            # consistent handling of sinking phases.
            bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")

        # Single-point upstream weighting via the shared, structure-caching helper (the
        # same extraction used by HUpwind). The upwind sparsity pattern is
        # flow-independent (the helper keeps both neighbour entries per face -- weight 1
        # upstream, explicit 0 downstream -- so the product is unchanged), which lets
        # re-discretization on a fixed mesh rewrite only the matrix data instead of
        # rebuilding CSRs. This is the dominant cost when the upwind runs per subdomain
        # on every nonlinear iteration. Bit-identical values to the explicit build; the
        # pattern carries extra structural zeros.
        cache = data.setdefault("_upwind_fast_cache", {}).setdefault(self.keyword, {})
        upwind, bound_dir, bound_neu = _single_point_upwind_matrices(
            sd,
            parameter_dictionary[self._flux_array_key],
            bc,
            num_components,
            cache=cache,
        )
        matrix_dictionary[self.upwind_matrix_key] = upwind
        matrix_dictionary[self.bound_transport_dir_matrix_key] = bound_dir
        matrix_dictionary[self.bound_transport_neu_matrix_key] = bound_neu

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

        # Re-discretization on a fixed mesh only changes the upwind DATA (flow signs).
        # The trace/inv-trace projections and mortar identity are pure geometry, and the
        # diagonal sparsity is invariant (_fixed_diag). Cache the geometry per
        # interface+keyword and rewrite the cached diagonals' .data in place on later
        # calls, instead of allocating fresh CSRs -- the dominant re-discretization cost
        # on many-interface problems. Bit-identical to the explicit build.
        cache = data_intf.setdefault("_upwind_coupling_fast_cache", {}).setdefault(
            self.keyword, {}
        )
        if "inv_trace" not in cache:
            # Mapping from upper dim cells to faces. The mortars always point from upper
            # to lower, so no sign flips; faces not adjacent to the mortar grid are
            # killed later by mortar projections.
            inv_trace_h = np.abs(sd_primary.divergence(dim=1))
            cache["inv_trace"] = inv_trace_h
            cache["trace"] = inv_trace_h.T  # trace-like projection from cells to faces
            cache["mortar_discr"] = sps.eye(intf.num_cells)
        matrix_dictionary[self.inv_trace_primary_matrix_key] = cache["inv_trace"]
        matrix_dictionary[self.trace_primary_matrix_key] = cache["trace"]
        matrix_dictionary[self.mortar_discr_matrix_key] = cache["mortar_discr"]

        # Find upwind weighting. if flag is True we use the upper weights if flag is
        # False we use the lower weights. Full diagonals keep the pattern fixed across
        # flow reversals, see _fixed_diag.
        flag = (lam_flux > 0).astype(float)
        not_flag = 1 - flag
        diags = cache.get("diags")
        if diags is None:
            upwind_from_primary = _fixed_diag(flag)
            upwind_from_secondary = _fixed_diag(not_flag)
            flux = _fixed_diag(lam_flux)
            cache["diags"] = (upwind_from_primary, upwind_from_secondary, flux)
        else:
            upwind_from_primary, upwind_from_secondary, flux = diags
            upwind_from_primary.data[:] = flag
            upwind_from_secondary.data[:] = not_flag
            flux.data[:] = lam_flux

        matrix_dictionary[self.upwind_primary_matrix_key] = upwind_from_primary
        matrix_dictionary[self.upwind_secondary_matrix_key] = upwind_from_secondary
        matrix_dictionary[self.flux_matrix_key] = flux

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
    """Dirichlet boundary faces that are inflow for the given flux direction. These
    are dropped from the transport matrix and handled by the boundary term."""
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

    Extraction of :meth:`Upwind.discretize`, reused for the two directions of
    :class:`HUpwind`. Returns ``(upwind, bound_transport_dir, bound_transport_neu)``.

    The sparsity pattern is flow-independent, so with a caller-owned ``cache`` dict
    re-discretization only rewrites the matrix data. The structure is rebuilt when a
    Dirichlet face flips inflow/outflow. The fast path applies to
    ``num_components == 1`` only and is bit-identical to the full build.
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

    # Data-only fast path: cached structure, single component, unchanged drop-mask.
    st = cache.get("struct") if (cache is not None and num_components == 1) else None
    if st is not None:
        if not st["has_dir"]:
            # No Dirichlet faces, so the drop-mask is fixed.
            bc_dir = st["bc_dir"]
            fast = True
        else:
            inflow = _dirichlet_inflow(bc, pos_flux, neg_flux, st["cf_dense"])
            drop = st["drop_neu"].copy()
            drop[inflow] = True
            fast = bool(np.array_equal(st["col_ok"] & ~drop[st["row"]], st["keep"]))
            if fast:
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
            # Cached indices are already sorted.
            upwind.has_sorted_indices = True
            return upwind, bc_dir, st["bc_neu"]

    # Full build: first call, changed structure, no cache, or num_components != 1.
    cf_dense = sd.cell_faces_as_dense()

    neumann_ind = np.where(bc.is_neu)[0]
    inflow_ind = _dirichlet_inflow(bc, pos_flux, neg_flux, cf_dense)
    drop_face = np.zeros(sd.num_faces, dtype=bool)
    drop_face[np.r_[neumann_ind, inflow_ind]] = True

    # Fixed-sparsity upwinding: each face keeps entries for both neighbour cells,
    # weight 1 upstream and an explicit 0 downstream. The pattern is purely geometric
    # and survives flow-direction flips; the product is unchanged.
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
            # Capture the COO -> sorted-CSR permutation, so later calls can do
            # upstream_mat.data = values[data_src] directly.
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
    """Two-direction upwinding for the buoyancy term.

    Stores two direction arrays and builds one single-point upwind matrix (plus
    boundary matrices) per direction: ``upwind_gamma`` upstream by ``gamma_flux``,
    ``upwind_delta`` upstream by ``delta_flux``. For hybrid upwinding the two
    directions are the inter-phase gravity flux with opposite signs. See
    :class:`~porepy.numerics.ad.discretizations.HUpwindAd` for the AD wrapper.
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

        gamma_dir = np.asarray(parameter_dictionary[self._gamma_flux_key])
        delta_dir = np.asarray(parameter_dictionary[self._delta_flux_key])
        # Per-direction structure caches, persisted in the per-grid data dict and
        # keyed by this discretization's keyword to avoid collisions.
        fast = data.setdefault("_hu_upwind_fast_cache", {})
        up_g, dir_g, neu_g = _single_point_upwind_matrices(
            sd,
            gamma_dir,
            bc,
            num_components,
            cache=fast.setdefault(self.keyword + ":gamma", {}),
        )
        up_d, dir_d, neu_d = _single_point_upwind_matrices(
            sd,
            delta_dir,
            bc,
            num_components,
            cache=fast.setdefault(self.keyword + ":delta", {}),
        )
        matrix_dictionary["transport_gamma"] = up_g
        matrix_dictionary["rhs_dir_gamma"] = dir_g
        matrix_dictionary["rhs_neu_gamma"] = neu_g
        matrix_dictionary["transport_delta"] = up_d
        matrix_dictionary["rhs_dir_delta"] = dir_d
        matrix_dictionary["rhs_neu_delta"] = neu_d


class HUpwindCoupling(UpwindCoupling):
    """Interface (mortar) counterpart of :class:`HUpwind`.

    Builds the mortar upwind matrices and signed flux for both directions in one
    :meth:`discretize`; the geometric trace matrices are built once and shared.
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

        # Re-discretization on a fixed mesh only changes the upwind DATA (flow signs);
        # the geometry (trace/inv_trace/mortar identity) and the diagonal sparsity
        # pattern are invariant. Cache them per interface+keyword and, on later calls,
        # rewrite the cached diagonals' .data in place instead of allocating fresh CSRs,
        # the dominant cost of re-discretizing a many-interface (mixed-dimensional)
        # problem. Bit-identical to the plain build below (same values, same fixed
        # _fixed_diag structure).
        cache = data_intf.setdefault("_hu_coupling_fast_cache", {}).setdefault(
            self.keyword, {}
        )
        if "inv_trace" not in cache:
            inv_trace_h = np.abs(sd_primary.divergence(dim=1))
            cache["inv_trace"] = inv_trace_h
            cache["trace"] = inv_trace_h.T
            cache["mortar_discr"] = sps.eye(intf.num_cells)
        matrix_dictionary["inv_trace"] = cache["inv_trace"]
        matrix_dictionary["trace"] = cache["trace"]
        matrix_dictionary["mortar_discr"] = cache["mortar_discr"]

        for suffix, key in (
            ("gamma", self._gamma_flux_key),
            ("delta", self._delta_flux_key),
        ):
            lf = np.sign(parameter_dictionary[key])
            flag = (lf > 0).astype(float)
            # Full diagonals keep the pattern fixed across flow reversals, see
            # _fixed_diag.
            diags = cache.get(suffix)
            if diags is None:
                up_p, up_s, flux = (
                    _fixed_diag(flag),
                    _fixed_diag(1.0 - flag),
                    _fixed_diag(lf),
                )
                cache[suffix] = (up_p, up_s, flux)
            else:
                up_p, up_s, flux = diags
                up_p.data[:] = flag
                up_s.data[:] = 1.0 - flag
                flux.data[:] = lf
            matrix_dictionary[f"upwind_primary_{suffix}"] = up_p
            matrix_dictionary[f"upwind_secondary_{suffix}"] = up_s
            matrix_dictionary[f"flux_{suffix}"] = flux
