"""Implementation of the multi-point flux approximation O-method.

"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp


class Mpfa(pp.FVElliptic):
    """Implementation of the multi-point flux approximation O-method.

    The method can be used directly on a single grid, by calling ``:meth:discretize``
    followed by ``:meth:assemble_matrix_rhs``, which will produce a linear system
    corresponding to the discretization of the second order elliptic equation, to be
    solved for the cell center values of the pressure.

    The method can also be used as a discretization method for a mixed-dimensional
    problem, but it is recommended to do so through the Ad framework, specifically via
    the wrapper class ``:class:porepy.numerics.ad.discretization.MpfaAd``. Likewise, the
    Ad framework is the recommended way to use the method for multiphysics problems, on
    both single and mixed-dimensional grids.

    The discretization is stored as a set of matrices in the dictionary
    ``data[pp.DISCRETIZATION_MATRICES][self.keyword]``, see ``:meth:discretize`` for
    details. The most important of these are the ``flux`` matrix, which is the
    transmissibility matrix, and the ``bound_flux`` matrix, which is the discretization
    of the boundary conditions. Documentation on how to use these matrices to assemble a
    linear system can be found ``:meth:assemble_matrix_rhs``. In addition, the
    discretization provides two matrices, ``bound_pressure_cell`` and
    ``bound_pressure_face`` that together form a reconstruction of the pressure trace at
    boundaries (external and internal). These can be computed by calling

    .. code-block:: Python
        bc_values = ... # Boundary condition values, can be a combination of Dirichlet
                        # and Neumann conditions

        p_cell_center = ... # Compute cell center pressures
        bound_pressure_cell = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.bound_pressure_cell_matrix_key
        ]  # Fetch the discretization matrix, see discretize() for details
        # Do the same for bound_pressure_face

        # Compute the pressure trace p_trace = bound_pressure_cell * p_cell_center +
        bound_pressure_face * bc_values

    This reconstruction can be decomposed into the value of the pressure in the cell
    next to the boundary face, and a correction term which consists of the pressure
    gradient dotted with the distance vector from the cell center to the boundary face.
    The cell center pressure is contained in the first term (``bound_pressure_cell *
    p_cell_center``), while the correction term is contained in both matrices, since
    the gradient is determined both by cell center values and boundary values.

    """

    def __init__(self, keyword: str) -> None:
        super(pp.Mpfa, self).__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """Discretize the second order elliptic equation using multi-point flux
        approximation.

        The method computes fluxes over faces in terms of pressures defined at the cell
        centers.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in ``data[pp.PARAMETERS][self.keyword]``.
            matrix_dictionary, for storage of discretization matrices.
                Stored in ``data[pp.DISCRETIZATION_MATRICES][self.keyword]``

        parameter_dictionary contains the entries:
            - second_order_tensor (``class:~porepy.params.tensor.SecondOrderTensor``):
                Permeability defined cell-wise. This is the effective permeability; any
                scaling of the permeability (such as with fracture apertures) should be
                included in the permeability.
            - bc (``class:~porepy.params.bc.BoundaryCondition``): boundary conditions.
            - ambient_dimension (``int``): Optional. Ambient dimension, used in the
                discretization of vector source terms. Defaults to the dimension of the
                grid.
            - mpfa_eta (``float``): Optional. Range [0, 1). Location of pressure
                continuity point. If not given, porepy tries to set an optimal value.
            - mpfa_inverter (``str``): Optional. Inverter to apply for local problems.
                Can take values 'python' (default), or 'numba'.
            - partition_arguments (``dict``): Optional. Arguments to control the number
                of subproblems used to discretize the grid. Can be either the target
                maximal memory use (controlled by keyword 'max_memory' in
                ``partition_arguments``), or the number of subproblems (keyword
                'num_subproblems' in ``partition_arguments``). If none are given, the
                default is to use 1e9 bytes of memory per subproblem. If both are given,
                the maximal memory use is prioritized.

        matrix_dictionary will be updated with the following entries:
            - ``flux: sps.csc_matrix (sd.num_faces, sd.num_cells)``
                flux discretization, cell center contribution
            - ``bound_flux: ps.csc_matrix (sd.num_faces, sd.num_faces)``
                flux discretization, face contribution
            - ``bound_pressure_cell: sps.csc_matrix (sd.num_faces, sd.num_cells)``
                Operator for reconstructing the pressure trace. Cell center contribution
            - ``bound_pressure_face: sps.csc_matrix (sd.num_faces, sd.num_faces)``
                Operator for reconstructing the pressure trace. Face contribution
            - ``vector_source: sps.csc_matrix (sd.num_faces, sd.num_cells * dim)``
                Discretization of the flux due to vector source term, cell center
                contribution.

        Parameters:
            sd: The grid to be discretized.
            data: The dictionary containing relevant information. See above for a
                description of required fields and storage of computed values.

        """
        parameter_dictionary: dict = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]
        # Dimension for vector source term field. Defaults to the same as the grid.
        # For grids embedded in a higher dimension, this must be set to the ambient
        # dimension.
        vector_source_dim: int = parameter_dictionary.get("ambient_dimension", sd.dim)

        # Short cut: if sd.dim == 0, construct empty matrices right away
        if sd.dim == 0:
            matrix_dictionary[self.flux_matrix_key] = sps.csc_matrix(
                (sd.num_faces, sd.num_cells)
            )
            matrix_dictionary[self.bound_flux_matrix_key] = sps.csc_matrix(
                (sd.num_faces, sd.num_faces)
            )
            matrix_dictionary[self.bound_pressure_cell_matrix_key] = sps.csc_matrix(
                (sd.num_faces, sd.num_cells)
            )
            matrix_dictionary[self.bound_pressure_face_matrix_key] = sps.csc_matrix(
                (sd.num_faces, sd.num_faces)
            )
            matrix_dictionary[self.vector_source_matrix_key] = sps.csc_matrix(
                (sd.num_faces, sd.num_cells * vector_source_dim)
            )
            matrix_dictionary[
                self.bound_pressure_vector_source_matrix_key
            ] = sps.csc_matrix((sd.num_faces, sd.num_cells * vector_source_dim))
            # Done
            return

        k: pp.SecondOrderTensor = parameter_dictionary["second_order_tensor"]
        bnd: pp.BoundaryCondition = parameter_dictionary["bc"]

        eta: Optional[float] = parameter_dictionary.get("mpfa_eta", None)
        inverter: Literal["numba", "python"] = parameter_dictionary.get(
            "mpfa_inverter", "python"
        )

        # Control of the number of subdomanis.
        max_memory, num_subproblems = pp.fvutils.parse_partition_arguments(
            parameter_dictionary.get("partition_arguments", {})
        )

        # Whether to update an existing discretization, or construct a new one.
        # If True, either specified_cells, _faces or _nodes should also be given, or
        # else a full new discretization will be computed
        update: bool = parameter_dictionary.get("update_discretization", False)

        # NOTE: active_faces are all faces to have their stencils updated, while
        # active_cells may form a larger set (to accurately update all faces on a
        # subgrid, it is necessary to assign some overlap in terms of cells).
        active_cells, active_faces = pp.fvutils.find_active_indices(
            parameter_dictionary, sd
        )

        if active_cells.size < sd.num_cells:
            # Extract a grid, and get global indices of its active faces and nodes.
            active_grid, extracted_faces, _ = pp.partition.extract_subgrid(
                sd, active_cells
            )
            # Constitutive law and boundary condition for the active grid.
            active_constit: pp.SecondOrderTensor = self._constit_for_subgrid(
                k, active_cells
            )

            # Extract the relevant part of the boundary condition.
            active_bound: pp.BoundaryCondition = self._bc_for_subgrid(
                bnd, active_grid, extracted_faces, sd
            )
        else:
            # The active grid is simply the grid
            active_grid = sd
            extracted_faces = active_faces
            active_constit = k
            active_bound = bnd

        # Bookkeeping.
        nf = active_grid.num_faces
        nc = active_grid.num_cells

        # To limit the memory need of discretization, we will split the discretization
        # into sub-problems, discretize these separately, and glue together the results.
        # This procedure requires some bookkeeping, in particular to keep track of how
        # many times a face has been discretized (faces may be shared between subgrids,
        # thus discretized multiple times).
        faces_in_subgrid_accum = []
        # Find an estimate of the peak memory need. This is used to decide how many
        # subproblems to split the discretization into.
        peak_memory_estimate = self._estimate_peak_memory(active_grid)

        # Empty matrices for flux, bound_flux and boundary pressure reconstruction. Will
        # be expanded as we go.
        # Implementation note: It should be relatively straightforward to
        # estimate the memory need of pressures (face_nodes -> node_cells -> unique).
        active_flux = sps.csr_matrix((nf, nc))
        active_bound_flux = sps.csr_matrix((nf, nf))
        active_bound_pressure_cell = sps.csr_matrix((nf, nc))
        active_bound_pressure_face = sps.csr_matrix((nf, nf))

        # For the vector sources, some extra care is needed in the projections between
        # local and global discretization matrices. The variable cell_vector_dim is used
        # at various places below.
        if sd.dim == 2:
            # In the 2d case, the discretization is done in 2d. This is compensated for
            # by the rotation towards the end of this function.
            cell_vector_dim: int = 2
        else:
            # Else, the discretization of the vector source is in vector_source_dim.
            # Exception: vector_source_dim = 0, assumedly for no assigned ambient
            #  dimension and sd.dim = 0. Then we need matrices of shape=[1,1]
            cell_vector_dim = max(1, vector_source_dim)

        active_vector_source = sps.csr_matrix((nf, nc * cell_vector_dim))
        active_bound_pressure_vector_source = sps.csr_matrix((nf, nc * cell_vector_dim))

        # Loop over all partition regions, construct local problems, and transfer
        # discretization to the entire active grid.
        for reg_i, (sub_sd, faces_in_subgrid, _, l2g_cells, l2g_faces) in enumerate(
            pp.fvutils.subproblems(
                active_grid, peak_memory_estimate, max_memory, num_subproblems
            )
        ):
            # The partitioning into subgrids is done with an overlap (see
            # fvutils.subproblems for a description). Cells and faces in the overlap
            # will have a wrong discretization in one of two ways: Those faces that are
            # strictly in the overlap should not be included in the current
            # sub-discretization (their will be in the interior of a different
            # subdomain). This contribution will be deleted locally, below. Faces that
            # are on the boundary between two subgrids will be discretized twice. This
            # will be handled by dividing the discretization by two. We cannot know
            # which faces are on the boundary (or perhaps we could, but that would
            # require more advanced bookkeeping), so we count the number of times a face
            # has been discretized, and divide by that number at the end (after having
            # iterated over all subdomains).
            #
            # Take note of which faces are discretized in this subgrid. Note that this
            # needs to use faces_in_subgrid, not l2g_faces, since the latter contains
            # faces in the overlap.
            faces_in_subgrid_accum.append(faces_in_subgrid)

            # Copy permeability tensor, and restrict to local cells
            loc_c: pp.SecondOrderTensor = self._constit_for_subgrid(
                active_constit, l2g_cells
            )

            # Boundary conditions are slightly more complex. Find local faces that are
            # on the global boundary. Then transfer boundary condition on those faces.
            loc_bnd: pp.BoundaryCondition = self._bc_for_subgrid(
                active_bound, sub_sd, l2g_faces, active_grid
            )

            # Eta can either be a scalar or a vector. If a vector valued eta is passed,
            # its length should be adjusted to match the number of subfaces in the
            # partitioned subgrid.
            if isinstance(eta, np.ndarray):
                loc_eta = pp.fvutils.adjust_eta_length(
                    eta=eta, sub_sd=sub_sd, l2g_faces=l2g_faces
                )

            # Non-array eta suggests eta is scalar. Thus no changes happen to eta.
            else:
                loc_eta = eta

            # Discretization of sub-problem
            discr_fields = self._flux_discretization(
                sub_sd,
                loc_c,
                loc_bnd,
                eta=loc_eta,
                inverter=inverter,
                ambient_dimension=vector_source_dim,
            )

            # Eliminate contribution from faces already discretized (the dual grids /
            # interaction regions may be structured so that some faces have previously
            # been partially discretized even if it has not been their turn until now)
            eliminate_face = np.where(
                np.logical_not(np.in1d(l2g_faces, faces_in_subgrid))
            )[0]
            pp.fvutils.remove_nonlocal_contribution(eliminate_face, 1, *discr_fields)

            # Split the discretization.
            (
                loc_flux,
                loc_bound_flux,
                loc_bound_pressure_cell,
                loc_bound_pressure_face,
                loc_vector_source,
                loc_bound_pressure_vector_source,
            ) = discr_fields

            # Next, transfer discretization matrices from the local to the active grid
            if active_grid.num_faces == faces_in_subgrid.size:
                # If all faces in the grid are truly within this subgrid (identified by
                # faces_in_subgrid, which does not include faces in the overlap), we
                # know the domain was not split, thus we need not bother with mappings.
                active_flux = loc_flux
                active_bound_flux = loc_bound_flux
                active_bound_pressure_cell = loc_bound_pressure_cell
                active_bound_pressure_face = loc_bound_pressure_face
                active_vector_source = loc_vector_source
                active_bound_pressure_vector_source = loc_bound_pressure_vector_source
            else:
                # Get a mapping from the local to the active grid
                face_map, cell_map = pp.fvutils.map_subgrid_to_grid(
                    active_grid, l2g_faces, l2g_cells, is_vector=False
                )
                # Update discretization on the active grid.
                active_flux += face_map * loc_flux * cell_map
                active_bound_flux += face_map * loc_bound_flux * face_map.transpose()

                # Update global face fields.
                active_bound_pressure_cell += (
                    face_map * loc_bound_pressure_cell * cell_map
                )
                active_bound_pressure_face += (
                    face_map * loc_bound_pressure_face * face_map.transpose()
                )
                # The vector source is a field of dimension vector_source_dim, and must
                # be mapped accordingly.
                _, cell_map_vec = pp.fvutils.map_subgrid_to_grid(
                    active_grid,
                    l2g_faces,
                    l2g_cells,
                    is_vector=True,
                    nd=cell_vector_dim,
                )
                active_vector_source += face_map * loc_vector_source * cell_map_vec
                active_bound_pressure_vector_source += (
                    face_map * loc_bound_pressure_vector_source * cell_map_vec
                )

        # Divide by the number of times a face has been discretized. This is necessary
        # to avoid double counting of faces on the boundary between subproblems. Note
        # that this is done before mapping from the active to the full grid, since the
        # subgrids (thus face map) was computed on the active grid.
        num_face_repetitions = np.bincount(np.concatenate(faces_in_subgrid_accum))

        scaling = sps.dia_matrix((1.0 / num_face_repetitions, 0), shape=(nf, nf))
        # All the discretization matrices must be updated.
        active_flux = scaling @ active_flux
        active_bound_flux = scaling @ active_bound_flux
        active_bound_pressure_cell = scaling @ active_bound_pressure_cell
        active_bound_pressure_face = scaling @ active_bound_pressure_face
        active_vector_source = scaling @ active_vector_source
        active_bound_pressure_vector_source = (
            scaling @ active_bound_pressure_vector_source
        )

        # We have reached the end of the discretization, what remains is to map the
        # discretization back from the active grid to the entire grid.
        if (
            active_grid.num_cells == sd.num_cells
            and active_grid.num_faces == sd.num_faces
        ):
            # Shortcut
            flux_glob = active_flux
            bound_flux_glob = active_bound_flux
            bound_pressure_cell_glob = active_bound_pressure_cell
            bound_pressure_face_glob = active_bound_pressure_face
            # The vector source term has a vector-sized number of rows, and needs a
            # different cell_map. See explanation of cell map dimension above.

            vector_source_glob = active_vector_source
            bound_pressure_vector_source_glob = active_bound_pressure_vector_source
        else:
            face_map, cell_map = pp.fvutils.map_subgrid_to_grid(
                sd, extracted_faces, active_cells, is_vector=False
            )

            # Update global face fields.
            flux_glob = face_map * active_flux * cell_map
            bound_flux_glob = face_map * active_bound_flux * face_map.transpose()
            bound_pressure_cell_glob = face_map * active_bound_pressure_cell * cell_map
            bound_pressure_face_glob = (
                face_map * active_bound_pressure_face * face_map.transpose()
            )
            # The vector source term has a vector-sized number of rows, and needs a
            # different cell_map. See explanation of cell map dimension above.
            _, cell_map_vec = pp.fvutils.map_subgrid_to_grid(
                sd, extracted_faces, active_cells, is_vector=True, nd=cell_vector_dim
            )
            vector_source_glob = face_map * active_vector_source * cell_map_vec
            bound_pressure_vector_source_glob = (
                face_map * active_bound_pressure_vector_source * cell_map_vec
            )

        # The vector source term was computed in the natural coordinates of the grid. If
        # the grid is embedded in a higher dimension, the source term will have the size
        # of that dimensions, and the discretization matrix should be adjusted
        # accordingly. The dimension of the vector source term is given by the parameter
        # vector_source_dim.
        # If the dimension of the grid is 3 (i.e. the maximum possible) or 0 (no vectors
        # are possible), there is no need for adjustment. Likewise, if sd.dim == 1, the
        # discretization was done with tpfa, and the vector_source matrix is already in
        # the full coordinates.

        # What is left is to fix the projection if sd.dim == 2. This is the part of the
        # code that requires the special sd.dim == 2 when cell_vector_dim is set above.
        if sd.dim == 2:
            # Use the same mapping of the geometry as was done in
            # self._flux_discretization(). This mapping is deterministic, thus the
            # rotation matrix should be the same as applied before. In this case, we
            # only need the rotation, and the active dimensions
            *_, R, dim, _ = pp.map_geometry.map_grid(sd)

            # We need to pick out the parts of the rotation matrix that gives the
            # in-plane (relative to the grid) parts, and apply this to all cells in the
            # grid. The simplest way to do this is to expand R[dim] via sps.block_diags,
            # however this scales poorly with the number of blocks. Instead, use the
            # existing workaround to create a csr matrix based on R, and then pick out
            # the right parts of that one.
            full_rot_mat = pp.matrix_operations.csr_matrix_from_blocks(
                # Replicate R with the right ordering of data elements
                np.tile(R.ravel(), (1, sd.num_cells)).ravel(),
                # size of the blocks - this will always be the dimension of the rotation
                # matrix, that is, 3 - due to the grid coordinates being 3d. If the
                # ambient dimension really is 2, this is adjusted below.
                3,
                # Number of blocks
                sd.num_cells,
            )
            # Get the right components of the rotation matrix.
            dim_expanded = np.where(dim)[0].reshape(
                (-1, 1)
            ) + vector_source_dim * np.array(np.arange(sd.num_cells))
            # Dump the irrelevant rows of the global rotation matrix.
            glob_R = full_rot_mat[dim_expanded.ravel("F")]

            # If the grid is truly 2d, we also need to dump the irrelevant columns of
            # the rotation matrix
            if sd.dim == vector_source_dim:
                glob_R = glob_R[:, dim_expanded.ravel("F")]

            # Append a mapping from the ambient dimension onto the plane of this grid
            vector_source_glob *= glob_R
            bound_pressure_vector_source_glob *= glob_R

        eliminate_faces = np.setdiff1d(np.arange(sd.num_faces), active_faces)
        pp.fvutils.remove_nonlocal_contribution(
            eliminate_faces,
            1,
            flux_glob,
            bound_flux_glob,
            bound_pressure_cell_glob,
            bound_pressure_face_glob,
            vector_source_glob,
            bound_pressure_vector_source_glob,
        )

        if update:
            matrix_dictionary[self.flux_matrix_key][active_faces] = flux_glob[
                active_faces
            ]
            matrix_dictionary[self.bound_flux_matrix_key][
                active_faces
            ] = bound_flux_glob[active_faces]
            matrix_dictionary[self.bound_pressure_cell_matrix_key][
                active_faces
            ] = bound_pressure_cell_glob[active_faces]
            matrix_dictionary[self.bound_pressure_face_matrix_key][
                active_faces
            ] = bound_pressure_face_glob[active_faces]

            matrix_dictionary[self.vector_source_matrix_key][
                active_faces
            ] = vector_source_glob[active_faces]
            matrix_dictionary[self.bound_pressure_vector_source_matrix_key][
                active_faces
            ] = bound_pressure_vector_source_glob[active_faces]

        else:
            matrix_dictionary[self.flux_matrix_key] = flux_glob
            matrix_dictionary[self.bound_flux_matrix_key] = bound_flux_glob
            matrix_dictionary[
                self.bound_pressure_cell_matrix_key
            ] = bound_pressure_cell_glob
            matrix_dictionary[
                self.bound_pressure_face_matrix_key
            ] = bound_pressure_face_glob
            matrix_dictionary[self.vector_source_matrix_key] = vector_source_glob
            matrix_dictionary[
                self.bound_pressure_vector_source_matrix_key
            ] = bound_pressure_vector_source_glob

    def update_discretization(self, sd: pp.Grid, data: dict) -> None:
        """Update discretization.

        The updates can generally come as a combination of two forms:
            1) The discretization on part of the grid should be recomputed.
            2) The old discretization can be used (in parts of the grid), but the
               numbering of unknowns has changed, and the discretization should be
               reorder accordingly.

        Information on the basis for the update should be stored in a field

            data['update_discretization']

        This should be a dictionary which could contain keys:

            modified_cells, modified_faces

        These define cells, faces and nodes that have been modified (either parameters,
        geometry or topology), and should be rediscretized. It is up to the
        discretization method to implement the change necessary by this modification.
        Note that depending on the computational stencil of the discretization method, a
        grid quantity may be rediscretized even if it is not marked as modified.

        The dictionary data['update_discretization'] should further have keys:

            cell_index_map, face_index_map

        these should specify sparse matrices that maps old to new indices. If not
        provided, the cell and face bookkeeping will be assumed constant.

        It is up to the caller to specify which parts of the grid to recompute, and
        how to update the numbering of degrees of freedom. If the discretization
        method does not provide a tailored implementation for update, it is not
        necessary to provide this information.

        Parameters:
            g: Grid to be rediscretized.
            data: With discretization parameters. See above for more information.

        """
        # Discretizations to be interpreted as scalar right quantities
        scalar_cell_right = [
            self.flux_matrix_key,
            self.bound_pressure_cell_matrix_key,
        ]
        # Keywords that should be interpreted as vector cell quantities
        vector_cell_right = [
            self.vector_source_matrix_key,
            self.bound_pressure_vector_source_matrix_key,
        ]

        scalar_face_right = [
            self.bound_flux_matrix_key,
            self.bound_pressure_face_matrix_key,
        ]

        # To the left, this is all face quantities
        scalar_face_left = [
            self.flux_matrix_key,
            self.bound_flux_matrix_key,
            self.vector_source_matrix_key,
            self.bound_pressure_vector_source_matrix_key,
            self.bound_pressure_cell_matrix_key,
            self.bound_pressure_face_matrix_key,
        ]

        # Dimension of vector quantities.
        vector_source_dim: int = data[pp.PARAMETERS][self.keyword].get(
            "ambient_dimension", sd.dim
        )
        pp.fvutils.partial_update_discretization(
            sd,
            data,
            self.keyword,
            self.discretize,
            dim=vector_source_dim,
            scalar_cell_right=scalar_cell_right,
            scalar_face_right=scalar_face_right,
            scalar_face_left=scalar_face_left,
            vector_cell_right=vector_cell_right,
        )

    def _flux_discretization(
        self,
        sd: pp.Grid,
        k: pp.SecondOrderTensor,
        bnd: pp.BoundaryCondition,
        inverter: Optional[Literal["python", "numba"]] = None,
        ambient_dimension: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> tuple[
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
    ]:
        """
        Actual implementation of the MPFA O-method. To calculate MPFA on a grid
        directly, either call this method, or, to respect the privacy of this method,
        the main mpfa method with no memory constraints.

        Method properties and implementation details.

        The pressure is discretized as a linear function on sub-cells (see reference
        paper). In this implementation, the pressure is represented by its cell center
        value and the sub-cell gradients (this is in contrast to most papers, which use
        auxiliary pressures on the faces; the current formulation is equivalent, but
        somewhat easier to implement). The method will give continuous fluxes over the
        faces, and pressure continuity for certain points (controlled by the parameter
        eta). This can be expressed as a linear system on the form

            (i)    A * grad_p              = 0
            (ii)   Ar * grad_p + Cr * p_cc = 0
            (iii)  B  * grad_p + C  * p_cc = 0
            (iv)   0             D  * p_cc = I

        Here, the first equation represents flux continuity, and involves only the
        pressure gradients (grad_p). The second equation gives the Robin conditions,
        relating flux to the pressure. The third equation gives pressure continuity over
        interior faces, thus B will contain distances between cell centers and the face
        continuity points, while C consists of +- 1 (depending on which side the cell is
        relative to the face normal vector). The fourth equation enforces the pressure
        to be unity in one cell at a time. Thus (i)-(iv) can be inverted to express the
        pressure gradients in terms of the cell center variables, that is, we can
        compute the basis functions on the sub-cells. Because of the method construction
        (again see reference paper), the basis function of a cell c will be non-zero on
        all sub-cells sharing a vertex with c. Finally, the fluxes as functions of cell
        center values are computed by insertion into Darcy's law (which is essentially
        half of A from (i), that is, only consider contribution from one side of the
        face. Boundary values can be incorporated with appropriate modifications -
        Neumann conditions will have a non-zero right hand side for (i), while Dirichlet
        gives a right hand side for (iii).

        Parameters:
            sd: The grid to be discretized. k: Permeability tensor. bnd: Boundary
            condition for the problem. inverter: Which method to use for inverting the
            local linear system in mpfa. ambient_dimension: Dimension of the full
            computational domain. Defaults to
                the dimension of ``sd``.
            eta: Location of the Mpfa continuity point. If not provided, it will be set
                to what is considered the best value for the given grid.

        Returns:
            A tuple of 6 elements

            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces, sd.num_cells))``

                Flux discretization matrix.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces, sd.num_faces))``

                Boundary condition discretization matrix.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces, sd.num_cells))``

                Matrix for cell-center potential contribution to face potential
                reconstruction.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces, sd.num_faces))``

                Matrix for boundary condition contribution to face potential
                reconstruction.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces,
                                                 ambient_dimension * sd.num_cells))``

                Matrix for vector source term potential contribution to face fluxes.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces,
                                                 ambient_dimension * sd.num_cells))``

                Matrix for vector source term contribution to face potential
                reconstruction.

        """

        if eta is None:
            eta = pp.fvutils.determine_eta(sd)
        if ambient_dimension is None:
            ambient_dimension = sd.dim

        # The method reduces to the more efficient TPFA in one dimension, so that method
        # may be called. In 0D, there is no internal discretization to be done.
        if sd.dim == 1:
            discr = pp.Tpfa(self.keyword)
            params = pp.Parameters(sd)
            params["bc"] = bnd
            params["second_order_tensor"] = k
            params["ambient_dimension"] = ambient_dimension

            d: dict = {
                pp.PARAMETERS: {self.keyword: params},
                pp.DISCRETIZATION_MATRICES: {self.keyword: {}},
            }
            discr.discretize(sd, d)
            matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][self.keyword]

            return (
                matrix_dictionary[self.flux_matrix_key],
                matrix_dictionary[self.bound_flux_matrix_key],
                matrix_dictionary[self.bound_pressure_cell_matrix_key],
                matrix_dictionary[self.bound_pressure_face_matrix_key],
                matrix_dictionary[self.vector_source_matrix_key],
                matrix_dictionary[self.bound_pressure_vector_source_matrix_key],
            )

        elif sd.dim == 0:
            return (
                sps.csr_matrix((0, sd.num_cells)),
                sps.csr_matrix((0, 0)),
                sps.csr_matrix((0, sd.num_cells)),
                sps.csr_matrix((0, 0)),
                sps.csr_matrix((0, sd.num_cells * max(ambient_dimension, 1))),
                sps.csr_matrix((0, sd.num_cells * max(ambient_dimension, 1))),
            )

        # The grid coordinates are always three-dimensional, even if the grid is really
        # 2D. This means that there is not a 1-1 relation between the number of
        # coordinates of a point / vector and the real dimension. This again violates
        # some assumptions tacitly made in the discretization (in particular that the
        # number of faces of a cell that meets in a vertex equals the grid dimension,
        # and that this can be used to construct an index of local variables in the
        # discretization). These issues should be possible to overcome, but for the
        # moment, we simply force 2D grids to be proper 2D.

        if sd.dim == 2:
            # Rotate the grid into the xy plane and delete third dimension. First make a
            # copy to avoid alterations to the input grid
            sd = sd.copy()
            (
                cell_centers,
                face_normals,
                face_centers,
                R,
                _,
                nodes,
            ) = pp.map_geometry.map_grid(sd)
            sd.cell_centers = cell_centers
            sd.face_normals = face_normals
            sd.face_centers = face_centers
            sd.nodes = nodes

            # Rotate the permeability tensor and delete last dimension
            k = k.copy()
            k.values = np.tensordot(R.T, np.tensordot(R, k.values, (1, 0)), (0, 1))
            k.values = np.delete(k.values, (2), axis=0)
            k.values = np.delete(k.values, (2), axis=1)

        # Define subcell topology, that is, the local numbering of faces, subfaces,
        # sub-cells and nodes. This numbering is used throughout the discretization.
        subcell_topology = pp.fvutils.SubcellTopology(sd)

        # Below, the boundary conditions should be defined on the subfaces.
        if bnd.num_faces == subcell_topology.num_subfno_unique:
            # The boundary conditions is already given on the subfaces
            subcell_bnd = bnd
            subface_rhs = True
        else:
            # If bnd is not already a sub-face_bound we extend it
            subcell_bnd = pp.fvutils.boundary_to_sub_boundary(bnd, subcell_topology)
            subface_rhs = False

        # The normal vectors used in the product are simply the face normals (with areas
        # downscaled to account for subfaces). The sign of nk_grad_all coincides with
        # the direction of the normal vector.
        (
            nk_grad_all,
            cell_node_blocks,
            sub_cell_index,
        ) = pp.fvutils.scalar_tensor_vector_prod(sd, k, subcell_topology)

        ## Contribution from subcell gradients to local system.
        # The pressure at a subface continuity point is given by the subcell pressure
        # plus a deviation computed as the distance from cell center to continuity point
        # times the (by construction constant)

        # Contribution from subcell gradient to pressure continuity condition.
        # The operation consists of two parts:
        # 1) The cell-face distances from subcells that share a subface are paired (put
        #    on the same row).
        # 2) The sign of the distance is changed for those subcells where the subface
        #    normal vector points into the cell. This happens both on interior subfaces,
        #    and on faces on the boundary.
        # NOTE: The second operation is reversed for Robin boundary conditions, see
        #       below.
        pr_cont_grad_paired = pp.fvutils.compute_dist_face_cell(
            sd, subcell_topology, eta
        )

        # Discretized Darcy's law: The flux over a subface is given by the area weighted
        # normal vector, multiplied with the subcell permeability, and expressed in
        # terms of the subcell gradient (to be computed). For interior subfaces, Darcy's
        # law is only computed for one adjacent subcell.
        # Minus sign to get flow from high to low potential.
        darcy = -nk_grad_all[subcell_topology.unique_subfno]

        # Pair fluxes over subfaces, that is, enforce conservation. The operation
        # consists of two parts:
        # 1) The expression nK computed from subcells that share a subface is paired
        #    (put on the same row). That is, the number of rows in nk_grad is reduced.
        # 2) The sign of the nK products is changed for those subcells where the subface
        #    normal vector points into the cell. This happens both on interior subfaces,
        #    and on faces on the boundary.
        nk_grad_paired = subcell_topology.pair_over_subfaces(nk_grad_all)

        ## Contribution from cell center potentials to local systems.

        # Cell center pressure contribution to flux continuity on subfaces: There is no
        # contribution (fluxes are driven by gradients only). Construct a zero matrix of
        # the right size, that is, a mapping from cells to subfaces.
        nk_cell = sps.coo_matrix(
            (np.zeros(1), (np.zeros(1), np.zeros(1))),
            shape=(subcell_topology.num_subfno, subcell_topology.num_cno),
        ).tocsr()

        # Cell center pressure contribution to pressure continuity on subfaces: The
        # contributions will be unity. For (sub)cells where the subface normal vector
        # points inwards, the contribution is -1, that is, we do the same modification
        # as for nk_grad_paired.

        # Recover the sign of the subface normal vectors relative to subcells (positive
        # if normal vector points out of subcell). The information can be obtained from
        # sd.cell_faces, which doubles as a discrete divergence operator. The .A suffix
        # is necessary to get a numpy array, instead of a scipy matrix.
        sgn = sd.cell_faces[subcell_topology.fno, subcell_topology.cno].A
        # Then insert the signs into a matrix that maps cells to subfaces
        pr_cont_cell_all = sps.coo_matrix(
            (sgn[0], (subcell_topology.subfno, subcell_topology.cno))
        ).tocsr()

        # Create a sign array that only contains one side (subcell) of each subface.
        # This will be used at various points below.
        sgn_unique = sd.cell_faces[
            subcell_topology.fno_unique, subcell_topology.cno_unique
        ].A.ravel("F")

        ## Discretize the Robin condition.
        # This takes the form
        #
        #   f + subcell_bnd.robin_weight * pressure_face * subface_area = something.
        #
        # The scaling is important here: The f is a flux integrated over the half face,
        # thus the scaling with the subface area is necessary.
        #
        # The pressure at face (pressure trace) is reconstructed from the cell center
        # pressure plus a deviation given by the cell-face distance times the subcell
        # gradient. For the local system, these terms contribute to cell center and
        # gradient terms, respectively.
        #
        # NOTE: In this implementation, the computation of the subface pressure replaces
        # the already discretized pressure continuity condition (the relevant rows are
        # modified). As part of the modification, the change of sign in
        # pr_cont_grad_paired and pr_cont_cell_all is reversed. That is, the cell
        # pressure is scaled by a positive number, while the gradient is multiplied with
        # the vector from cell center to face continuity point.

        # Compute the sign of the subcells relative to the normal vector of the subface.
        # The sign is scaled by the Robin weight, and the area of the subface.
        # The conditions for subfaces not on a Robin boundary will be eliminated
        # below (in computation of pr_cont_grad).
        num_nodes = np.diff(sd.face_nodes.indptr)
        sgn_scaled_by_subface_area = (
            subcell_bnd.robin_weight
            * sgn_unique
            * sd.face_areas[subcell_topology.fno_unique]
            / num_nodes[subcell_topology.fno_unique]
        )
        # pair_over_subfaces flips the sign so we flip it back.
        # Contribution from cell centers.
        pr_trace_grad_all = sps.diags(sgn_scaled_by_subface_area) * pr_cont_grad_paired
        # Contribution from gradient.
        pr_trace_cell_all = sps.coo_matrix(
            (
                subcell_bnd.robin_weight[subcell_topology.subfno]
                * sd.face_areas[subcell_topology.fno]
                / num_nodes[subcell_topology.fno],
                (subcell_topology.subfno, subcell_topology.cno),
            )
        ).tocsr()

        del sgn, sgn_scaled_by_subface_area

        # Mapping from sub-faces to faces
        hf2f = sps.coo_matrix(
            (
                np.ones(subcell_topology.unique_subfno.size),
                (subcell_topology.fno_unique, subcell_topology.subfno_unique),
            ),
            shape=(sd.num_faces, subcell_topology.num_subfno_unique),
        )

        # If the grid has a periodic boundary, the left faces are topologically
        # connected to the right faces. This is included in the SubcellTopology class by
        # merging the right faces with the left faces. This means that hf2f is a mapping
        # from topological unique half faces to topological unique faces. However, we
        # want the discretization to be applied to the topology of g (where the left and
        # right faces are different). We therefore map the left faces of the
        # SubcellTopology to the right faces of the grid:
        if hasattr(sd, "periodic_face_map"):
            indices = np.arange(sd.num_faces)
            # The left faces should be mapped to the right faces
            indices[sd.periodic_face_map[1]] = sd.periodic_face_map[0]
            indptr = np.arange(sd.num_faces + 1)
            data = np.ones(sd.num_faces, dtype=int)
            periodic2face = sps.csr_matrix(
                (data, indices, indptr), (sd.num_faces, sd.num_faces)
            )
            # Update hf2f so that it maps to the faces of sd.
            hf2f = periodic2face * hf2f

        # The boundary faces will have either a Dirichlet or Neumann condition, or Robin
        # condition. Obtain mappings to exclude boundary faces.
        bound_exclusion = pp.fvutils.ExcludeBoundaries(
            subcell_topology, subcell_bnd, sd.dim
        )

        # No flux conditions for Dirichlet and Robin boundary faces.
        nk_grad_n = bound_exclusion.exclude_robin_dirichlet(nk_grad_paired)
        nk_cell = bound_exclusion.exclude_robin_dirichlet(nk_cell)

        # Robin condition is only applied to Robin boundary faces.
        if bound_exclusion.any_rob:
            nk_grad_r = bound_exclusion.keep_robin(nk_grad_paired)
            pr_trace_grad = bound_exclusion.keep_robin(pr_trace_grad_all)
            pr_trace_cell = bound_exclusion.keep_robin(pr_trace_cell_all)

        else:
            # Make empty variables, just to have something to fill in the matrix (see
            # usage below).
            nk_grad_r = sps.csr_matrix((0, nk_grad_paired.shape[1]))
            pr_trace_grad = sps.csr_matrix((0, pr_trace_grad_all.shape[1]))
            pr_trace_cell = sps.csr_matrix((0, pr_trace_cell_all.shape[1]))

        # No pressure condition for Neumann or Robin boundary faces
        pr_cont_grad = bound_exclusion.exclude_neumann_robin(pr_cont_grad_paired)
        pr_cont_cell = bound_exclusion.exclude_neumann_robin(pr_cont_cell_all)

        # The discretization is now in a sense complete: We have discretized the flux
        # and pressure continuity equations, with the necessary adjustments for boundary
        # conditions. The resulting linear system takes the form
        #
        #  (nk_grad_n                , nk_cell  )
        #  (nk_grad_r - pr_trace_grad, -pr_trace_cell) (subcell_grad)    = RHS
        #  (pr_cont_grad             , pr_cont_cell  ) (center_pressure)
        #
        # The first equation enforces flux continuity (with nk_cell=0)
        # The second equation enforces Robin boundary conditions
        # The third equation enforces pressure continuity.
        # RHS is detailed below, this enforces calculation of the discrete basis
        # function.
        #
        # The next step is to compute the subcell gradients as a function of the
        # ultimate degrees of freedom in the discretization scheme, that is cell center
        # pressures and boundary conditions. For cell center pressures, assign a unit
        # pressure in one cell center at a time while assigning zero pressures in other
        # cells (this is implemented as moving the second column in the above matrix to
        # the right hand side). When solving for subcell_grad, this gives the gradients,
        # in effect, the boundary conditions.
        #
        # Implementation of boundary conditions need special mentioning:
        # First, the rows corresponding to boundary subfaces are modified:
        #
        #  i) On Neumann subfaces, only the first block row is kept, second and third
        #     row are eliminated (there is no Robin or pressure continuity condition).
        #  ii) Robin conditions have dedicated rows, these subfaces are eliminated from
        #      the first and third row.
        #  iii) For Dirichlet subfaces, only the third block row is kept.
        #
        # As an implementation note, the fastest approach found is to explicitly
        # calculate the inverse of the matrix
        #
        #  (nk_grad_n)
        #  (nk_grad_r - pr_trace_grad)
        #  (pr_cont_grad)
        #
        # The calculation exploits the block diagonal structure of the matrix (after a
        # permutation that gathers all columns and rows associated with a single vertex,
        # that is, an interaction region). Fast inverses can then be used. After
        # inversion, subscale gradients can be found for any right hand side by a
        # matrix-vector product.

        # System of equations for the subcell gradient variables. On block diagonal
        # form.
        num_nk_cell = nk_cell.shape[0]
        num_nk_rob = nk_grad_r.shape[0]
        num_pr_cont_grad = pr_cont_grad.shape[0]

        # Assemble and scale the full system.
        grad_eqs = sps.vstack((nk_grad_n, nk_grad_r - pr_trace_grad, pr_cont_grad))

        # To lower the condition number of the local linear systems, the equations that
        # represents flux and pressure continuity, as well as the Robin condition (which
        # is a combination) should have ideally have similar scaling. This is in general
        # not possible (for instance, anisotropic permeabilities almost inevitably leads
        # to some rows having elements that differ significantly in size). Nevertheless,
        # we try to achieve reasonably conditioned local problems. A simple approach has
        # turned out to give reasonable results: For all continuity conditions compute
        # the sum of the absolute values of non-zero elements and scale the entire row
        # with the inverse of the sum value. This is equivalent to diagonal left
        # preconditioner for the system. In order not to modify the solution, we will
        # also need to left precondition the right-hand side.
        #
        # IMPLEMENTATION NOTE: Experimentation with more advanced scalings turned out
        # not to give better results than the simple row sum approach.
        full_scaling = pp.fvutils.diagonal_scaling_matrix(grad_eqs)

        # Scale the matrix before inversion.
        grad_eqs = full_scaling * grad_eqs

        del nk_grad_n, nk_grad_r, pr_trace_grad

        # So far, the local numbering has been based on the numbering scheme implemented
        # in SubcellTopology (which treats one cell at a time). For efficient inversion
        # (below), it is desirable to get the system over to a block-diagonal structure,
        # with one block centered around each vertex. Obtain the necessary mappings.
        rows2blk_diag, cols2blk_diag, size_of_blocks = self._block_diagonal_structure(
            sub_cell_index,
            cell_node_blocks,
            subcell_topology.nno_unique,
            bound_exclusion,
        )

        # Re-organize system into a block-diagonal form.
        grad = rows2blk_diag * grad_eqs * cols2blk_diag

        del grad_eqs
        # Invert the system, and map back to the original form. Also right multiply with
        # the diagonal scaling matrix. Since we will multiply igrad with various right
        # hand side matrices to construct the discretizations, this is equivalent with
        # scaling the right hand side terms.
        igrad = (
            cols2blk_diag
            * pp.matrix_operations.invert_diagonal_blocks(
                grad, size_of_blocks, method=inverter
            )
            * rows2blk_diag
        ) * full_scaling

        del grad, cols2blk_diag, rows2blk_diag

        # Technical note: The elements in igrad are organized as follows: The fields
        # subcell_topology.cno and .nno will together identify Nd placements in igrad
        # that are associated with the same cell and the same node, that is, they belong
        # to the same subcell. These placements are used to store the discrete gradient
        # of that cell, with the first item representing the x-component etc.
        # As an example, to find the gradient in the subcell of cell ci, associated with
        # node ni, first find the indexes of subcell_topology.cno and .nno that contain
        # ci and ni, respectively. The first of these indexes give the row of the
        # x-component of the gradient, the second the y-component etc.
        #
        # The columns of igrad correspond to the ordering of the equations in grad; as
        # recovered in _block_diagonal_structure. In practice, the first columns
        # correspond to unit pressures assigned to faces (as used for boundary
        # conditions or to discretize discontinuities over internal faces, say, to
        # represent heterogeneous gravity), while the latter group gives gradients
        # induced by cell center pressures.
        #
        # ASSUMPTIONS: 1) Each cell has exactly Nd faces meeting in a vertex; or else,
        # there would not be an exact match between the number of equal (nno-cno) pairs
        # and the number of components in the gradient. This assumption is always okay
        # in 2d, in 3d it rules out cells shaped as pyramids, in which case mpfa is not
        # defined without making further specifications of the method.
        # 2) The number of components in the gradient is equal to the spatial dimension
        # of the grid, as defined in sd.dim. Thus 2d grids embedded in 3d will run into
        # trouble, unless the grid is first projected down to its natural plane. This
        # can be fixed by a more general implementation, but it would require quite deep
        # changes to the code.

        # Flux discretization: Negative in front of sps.vstack comes from moving the
        # cell center unknown in the discretized sytsem to the right hand side. The
        # negative in front of pr_trace_cell comes from the grad_eqs
        rhs_cells = -sps.vstack(
            (
                nk_cell,
                -pr_trace_cell,
                pr_cont_cell,
            )
        )
        # Compute matrix-matrix product, this is used in several steps below
        darcy_igrad = darcy * igrad

        flux = darcy_igrad * rhs_cells

        # Boundary conditions
        rhs_bound = self._create_bound_rhs(
            subcell_bnd,
            bound_exclusion,
            subcell_topology,
            sgn_unique,
            sd,
            num_nk_cell,
            num_nk_rob,
            num_pr_cont_grad,
            subface_rhs,
        )
        # Discretization of boundary values
        bound_flux = darcy_igrad * rhs_bound

        # Obtain the reconstruction of the pressure
        dist_cell, cell_centers = reconstruct_presssure(sd, subcell_topology, eta)
        # Compute matrix-matrix product, this is used in several steps below
        dist_cell_igrad = dist_cell * igrad

        pressure_trace_cell = dist_cell_igrad * rhs_cells + cell_centers
        pressure_trace_bound = dist_cell_igrad * rhs_bound

        area_scaling = 1.0 / (hf2f * np.ones(hf2f.shape[1]))
        area_mat = hf2f * sps.diags(hf2f.T * area_scaling)
        if not subface_rhs:
            # In this case we set the value at a face, thus, we need to distribute the
            # face values to the subfaces. We do this by an area-weighted average. The
            # flux will in this case be integrated over the faces, that is:
            #   flux *\cdot * normal * face_area
            bound_flux = hf2f * bound_flux * hf2f.T
            flux = hf2f * flux
            pressure_trace_bound = area_mat * pressure_trace_bound * hf2f.T
            pressure_trace_cell = area_mat * pressure_trace_cell

        # Also discretize vector source terms for Darcy's law. discr_div_vector_source
        # is the discretised vector source, which is interpreted as a force on a subface
        # due to imbalance in cell-center vector sources. This term is computed on a
        # sub-cell basis and has dimensions (num_subfaces, num_subcells * nd)
        discr_vector_source, vector_source_bound = self._discretize_vector_source(
            sd,
            subcell_topology,
            bound_exclusion,
            darcy_igrad,
            dist_cell_igrad,
            nk_grad_all,
            nk_grad_paired,
        )

        # Output should be on cell-level (not sub-cell)
        sc2c = pp.fvutils.cell_vector_to_subcell(
            sd.dim, sub_cell_index, cell_node_blocks[0]
        )

        vector_source = hf2f * discr_vector_source * sc2c
        bound_pressure_vector_source = area_mat * vector_source_bound * sc2c

        return (
            flux,
            bound_flux,
            pressure_trace_cell,
            pressure_trace_bound,
            vector_source,
            bound_pressure_vector_source,
        )

    def _discretize_vector_source(
        self,
        sd: pp.Grid,
        subcell_topology: pp.fvutils.SubcellTopology,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
        darcy_igrad: sps.spmatrix,
        dist_cell_igrad: sps.spmatrix,
        nk_grad_all: sps.spmatrix,
        nk_grad_paired: sps.spmatrix,
    ) -> tuple[sps.spmatrix, sps.spmatrix]:
        """Consistent discretization of the divergence of the vector source term
        in MPFA-O method.

        An example of a vector source is the gravitational forces in Darcy's law. For
        more details, see Starnoni et al (2020), Consistent discretization of flow for
        inhomogeneoug gravitational fields, WRR.

        Parameters:
            g: grid to be discretized
            subcell_topology: Wrapper class for numbering of subcell faces, cells
                etc.
            bound_exclusion: Object that can eliminate faces related to boundary
                conditions.
            darcy_igrad: Product of the discrete Darcy law with the basis functions
                (igrad)
            dist_igrad: Product of
            nk_grad_all: Product between nK products on a sub-cell level and the basis
                functions (igrad)
            nk_grad_paired: nK products after pairing

        Returns:
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces,
                                                 ambient_dimension * sd.num_cells))``

                Matrix for vector source term potential contribution to face fluxes.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces,
                                                 ambient_dimension * sd.num_cells))``

                Matrix for vector source term contribution to face potential
                reconstruction.

        """
        # Method properties and implementation details.
        # Basis functions, namely 'flux' and 'bound_flux', for the pressure
        # discretization are obtained as in standard MPFA-O method. Vector source is
        # represented as forces in the cells. However, jumps in vector sources over a
        # cell face act as flux imbalance, and thus induce additional pressure gradients
        # in the sub-cells. An additional system is set up, which applies non-zero
        # conditions to the flux continuity equation. This can be expressed as a linear
        # system on the form:
        #
        #     (i)   A * grad_p            = I
        #     (ii)  B * grad_p + C * p_cc = 0
        #     (iii) 0            D * p_cc = 0
        #
        # Thus (i)-(iii) can be inverted to express the additional pressure gradients
        # due to imbalance in vector sources as in terms of the cell center variables.
        # Thus we can compute the basis functions 'vector_source_jumps' on the
        # sub-cells. To ensure flux continuity, as soon as a convention is chosen for
        # what side the flux evaluation should be considered on, an additional term,
        # called 'vector_source_faces', is added to the full flux. This latter term
        # represents the flux due to cell-center vector source acting on the face from
        # the chosen side. The pair subfno_unique-unique_subfno gives the side
        # convention. The full flux on the face is therefore given by
        #
        # q = flux * p + bound_flux * p_b
        #     + (vector_source_jumps + vector_source_faces) * vector_source
        #
        # Output: vector_source = vector_source_jumps + vector_source_faces

        # The strategy is as follows.
        # 1. assemble r.h.s. for the new linear system, needed for the term
        #     'vector_source_jumps'
        # 2. compute term 'vector_source_faces'

        num_subfno = subcell_topology.num_subfno
        num_subfno_unique = subcell_topology.subfno_unique.size

        # Step 1
        # The vector source term in the flux continuity equation is discretized as a
        # force on the faces. The right hand side is thus formed of the unit vector.

        # Construct a mapping from all subfaces to those with flux conditions, that is,
        # all but Dirichlet boundary faces.
        # Identity matrix, one row per subface
        vals = np.ones(num_subfno_unique)
        I_subfno = sps.dia_matrix(
            (vals, 0), shape=(num_subfno_unique, num_subfno_unique)
        )

        flux_eq_map_internal_neumann = bound_exclusion.exclude_robin_dirichlet(I_subfno)
        # Exclude Dirichlet and Robin faces
        if bound_exclusion.any_rob:
            # Robin condition is only applied to Robin boundary faces
            flux_eq_map_robin = bound_exclusion.keep_robin(I_subfno)

            # The Robin condition is appended at the end of the flux balance equations
            # This map is thus applicable to all flux balance equations
            flux_eq_map = sps.vstack([flux_eq_map_internal_neumann, flux_eq_map_robin])
        else:
            # We can use the internal and Neumann faces
            flux_eq_map = flux_eq_map_internal_neumann

        # No right hand side for cell pressure equations.
        num_dir_subface = (
            bound_exclusion.exclude_neu_rob.shape[1]
            - bound_exclusion.exclude_neu_rob.shape[0]
        )
        # Size of the pressure continuity system
        num_zeros = num_subfno - num_dir_subface
        # Expand the flux_eq_map with zeros in the rows corresponding to the pressure
        # contiuity equations This should really be renamed, but resize only works in
        # place, and vstack or similar takes much more time
        new_shape = (flux_eq_map.shape[0] + num_zeros, flux_eq_map.shape[1])
        flux_eq_map.resize(new_shape)
        # Right hand side term is multiplied with -1
        rhs_map = -flux_eq_map

        # prepare for computation of imbalance coefficients, that is jumps in
        # cell-centers vector sources, ready to be  multiplied with inverse gradients.
        # This is the middle term on the right hand side in Eq (31) in Starnoni et al
        prod = rhs_map * nk_grad_paired
        vector_source_jumps = -darcy_igrad * prod
        # The vector source should have no impact on the boundary conditions (on
        # Dirichlet boundaries, the pressure is set directly, on Neumann, the flux set
        # is interpreted as a total flux). However, to recover the face pressure at a
        # Neumann boundary, the contribution from the vector source must be accounted
        # for. Referring to (27)-(28) in Starnoni et al, the variable prod can be
        # interpreted as giving the pressure gradients resulting from a unit vector
        # source (note that in a boundary cell, nk_grad_paired will only contain
        # contribution from the single neighboring cell). Multiply this by the distance
        # from the cell center to the continuity point to get the pressure offset.
        vector_source_bound = -dist_cell_igrad * prod

        # Step 2

        # Mapping from subface to unique subface for scalar problems. This mapping gives
        # the convention from which side the force should be evaluated on.
        map_unique_subfno = sps.coo_matrix(
            (
                np.ones(num_subfno_unique),
                (subcell_topology.subfno_unique, subcell_topology.unique_subfno),
            ),
            shape=(subcell_topology.num_subfno_unique, subcell_topology.fno.size),
        )

        # Prepare for computation of div_vector_source_faces term
        vector_source_faces = map_unique_subfno * nk_grad_all

        return (vector_source_jumps + vector_source_faces, vector_source_bound)

    """
    The functions below are helper functions, which are not really necessary to
    understand in detail to use the method. They also tend to be less well
    documented.
    """

    def _estimate_peak_memory(self, sd: pp.Grid) -> int:
        """Compute a rough estimate of peak memory need for the discretization.

        Parameters:
            sd: The grid to be discretized

        Returns:
            The estimated peak memory need.

        """
        nd = sd.dim
        if nd == 0:
            return 0

        num_cell_nodes = sd.cell_nodes().sum(axis=1)

        # Number of unknowns around a vertex: nd per cell that share the vertex for
        # pressure gradients, and one per cell (cell center pressure)
        num_grad_unknowns = nd * num_cell_nodes

        # The most expensive field is the storage of igrad, which is block diagonal with
        # num_grad_unknowns sized blocks
        igrad_size = num_grad_unknowns.sum()

        # The discretization of Darcy's law will require nd (that is, a gradient) per
        # sub-face.
        num_sub_face = sd.face_nodes.sum()
        darcy_size = nd * num_sub_face

        # Balancing of fluxes will require 2*nd (gradient on both sides) fields per
        # sub-face
        nk_grad_size = 2 * nd * num_sub_face
        # Similarly, pressure continuity requires 2 * (nd+1) (gradient on both sides,
        # and cell center pressures) numbers
        pr_cont_size = 2 * (nd + 1) * num_sub_face

        total_size = igrad_size + darcy_size + nk_grad_size + pr_cont_size

        # Not covered yet is various fields on subcell topology, mapping matrices
        # between local and block ordering etc.
        return total_size

    def _block_diagonal_structure(
        self,
        sub_cell_index: np.ndarray,
        cell_node_blocks: np.ndarray,
        nno: np.ndarray,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
    ) -> tuple[sps.spmatrix, sps.spmatrix, np.ndarray]:
        """Define matrices to turn linear system into block-diagonal form.

        Parameters
            sub_cell_index: Index of all subcells.
            cell_node_blocks: Pairs of cell and node pairs, which defines sub-cells.
            nno: Node numbers associated with balance equations
            bound_exclusion: Object to eliminate faces associated with different types
                of boundary conditions.

        Returns:
            :obj:`~scipy.sparse.spmatrix`: Transform rows of linear system to
                block-diagonal form.

            :obj:`~scipy.sparse.spmatrix`: Transform colmns of linear system to
                block-diagonal form.

            np.ndarray: Number of equations in each block.

        """
        # Stack node numbers of equations on top of each other, and sort them to get
        # block-structure. First eliminate node numbers at the boundary, where the
        # equations are either of flux, pressure continuity or Robin
        nno_flux = bound_exclusion.exclude_robin_dirichlet(nno)
        nno_pressure = bound_exclusion.exclude_neumann_robin(nno)
        # we have now eliminated all nodes related to robin, we therefore add them
        nno_rob = bound_exclusion.keep_robin(nno)

        node_occ = np.hstack((nno_flux, nno_rob, nno_pressure))
        sorted_ind = np.argsort(node_occ)
        sorted_nodes_rows = node_occ[sorted_ind]
        # Size of block systems
        size_of_blocks = np.bincount(sorted_nodes_rows.astype("int64"))
        rows2blk_diag = sps.coo_matrix(
            (np.ones(sorted_nodes_rows.size), (np.arange(sorted_ind.size), sorted_ind))
        ).tocsr()

        # cell_node_blocks[1] contains the node numbers associated with each sub-cell
        # gradient (and so column of the local linear systems). A sort of these will
        # give a block-diagonal structure
        sorted_nodes_cols = np.argsort(cell_node_blocks[1])
        subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel("F")
        cols2blk_diag = sps.coo_matrix(
            (
                np.ones(sub_cell_index.size),
                (subcind_nodes, np.arange(sub_cell_index.size)),
            )
        ).tocsr()

        return rows2blk_diag, cols2blk_diag, size_of_blocks

    def _create_bound_rhs(
        self,
        bnd: pp.BoundaryCondition,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
        subcell_topology: pp.fvutils.SubcellTopology,
        sgn: np.ndarray,
        sd: pp.Grid,
        num_flux: int,
        num_rob: int,
        num_pr: int,
        subface_rhs: bool,
    ) -> sps.spmatrix:
        """Define rhs matrix to get basis functions for incorporates boundary
        conditions.

        Parameters:
            bnd: Boundary condition object.
            bound_exclusion: Object to eliminate faces associated with different types
                of boundary conditions.
            subcell_topology: Object that carries information on the subcell topology.
            sgn: Sign of subfaces, used to define the direction of fluxes.
            sd: Grid to be discretized.
            num_flux: Number of equations for flux continuity.
            num_rob: Number of equations for Robin conditions.
            num_pr: Number of equations for pressure continuity.
            subface_rhs: If True, the rhs will be defined on subfaces. Otherwise, it
                will be defined on faces.

        Returns:
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces, sd.num_faces))``

                    Matrix that can be multiplied with inverse block matrix to get basis
                    functions for boundary values. If subface_rhs is True, the matrix will
                    have ``subcell_topology.num_subfno_unique`` faces.

        """
        # For primal-like discretizations like the MPFA, internal boundaries are handled
        # by assigning Neumann conditions.
        is_dir = np.logical_and(bnd.is_dir, np.logical_not(bnd.is_internal))
        is_neu = np.logical_or(bnd.is_neu, bnd.is_internal)
        is_rob = np.logical_and(bnd.is_rob, np.logical_not(bnd.is_internal))

        fno = subcell_topology.fno_unique
        num_neu = np.sum(is_neu)
        num_dir = np.sum(is_dir)
        if not num_rob == np.sum(is_rob):
            raise AssertionError()

        num_bound = num_neu + num_dir + num_rob

        # Neumann and Robin boundary conditions. Neumann and Robin conditions are
        # essentially the same for the rhs (the rhs for both is a flux). However, we
        # need to be carefull and get the indexing correct as seen from the local
        # system, that is, first Neumann, then Robin and last Dirichlet. Find Neumann
        # and Robin faces, exclude Dirichlet faces (since these are excluded from the
        # right hand side linear system), and do necessary formating.
        neu_ind = np.argwhere(
            bound_exclusion.exclude_robin_dirichlet(is_neu.astype("int64"))
        ).ravel("F")
        if bound_exclusion.any_rob:
            rob_ind = np.argwhere(
                bound_exclusion.keep_robin(is_rob.astype("int64"))
            ).ravel("F")
            neu_rob_ind = np.argwhere(
                bound_exclusion.exclude_dirichlet((is_rob + is_neu).astype("int64"))
            ).ravel("F")
        else:
            # Shortcut if no Robin conditions
            rob_ind = np.array([], dtype=np.int64)
            neu_rob_ind = neu_ind

        # We also need to map the respective Neumann, Robin, and Dirichlet half-faces to
        # the global half-face numbering (also interior faces). The latter should not
        # have Dirichlet and Neumann excluded (respectively), and thus we need new
        # fields.
        neu_ind_all = np.argwhere(is_neu.astype("int")).ravel("F")
        rob_ind_all = np.argwhere(is_rob.astype("int")).ravel("F")
        dir_ind_all = np.argwhere(is_dir.astype("int")).ravel("F")
        num_face_nodes = np.diff(sd.face_nodes.indptr)

        # We now merge the neuman and robin indices since they are treated equivalent.
        if rob_ind.size == 0:
            neu_rob_ind = neu_ind
        elif neu_ind.size == 0:
            neu_rob_ind = rob_ind + num_flux
        else:
            neu_rob_ind = np.hstack((neu_ind, rob_ind + num_flux))
        neu_rob_ind_all = np.hstack((neu_ind_all, rob_ind_all))

        # For the Neumann/Robin boundary conditions, we define the value as seen from
        # the innside of the domain. E.g. outflow is defined to be positive. We
        # therefore set the matrix indices to -1. We also have to scale it with the
        # number of nodes per face because the flux of a face is the sum of its
        # half-face fluxes.
        #
        # EK: My understanding of the multiple -1s in the flux equation for boundary
        # conditions:
        # 1) -nk_grad in the local system gets its sign changed if the boundary
        #    normal vector points into the cell.
        # 2) During global assembly, the flux matrix is hit by the divergence
        #    operator, which will give another -1 for cells with inwards pointing
        #    normal vector.
        if subface_rhs:
            # In this case we set the rhs for the sub-faces. Note that the rhs values
            # should be integrated over the subfaces, that is
            #       flux_neumann *\cdot * normal * subface_area
            scaled_sgn = -1 * np.ones(neu_rob_ind_all.size)
        else:
            # In this case we set the value at a face, thus, we need to distribute the
            # face values to the subfaces. We do this by an area-weighted average. Note
            # that the rhs values should in this case be integrated over the faces, that
            # is: flux_neumann *\cdot * normal * face_area
            scaled_sgn = -1 / num_face_nodes[fno[neu_rob_ind_all]]

        # Build up the boundary conditions. First for flux continuity equations
        if neu_rob_ind.size > 0:
            # Prepare for construction of a coo-matrix
            rows = neu_rob_ind
            cols = np.arange(neu_rob_ind.size)
            data = scaled_sgn
        else:
            # Special handling when no elements are found.
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)
            data = np.array([], dtype=int)

        # Dirichlet boundary conditions
        dir_ind = np.argwhere(
            bound_exclusion.exclude_neumann_robin(is_dir.astype("int64"))
        ).ravel("F")
        if dir_ind.size > 0:
            rows = np.hstack((rows, num_flux + num_rob + dir_ind))
            cols = np.hstack((cols, num_neu + num_rob + np.arange(dir_ind.size)))
            data = np.hstack((data, sgn[dir_ind_all]))
        else:
            # No need to do anything here
            pass

        # Number of elements in neu_ind and neu_ind_all are equal, we can test with any
        # of them. Same with dir.
        if neu_rob_ind.size > 0 and dir_ind.size > 0:
            neu_rob_dir_ind = np.hstack([neu_rob_ind_all, dir_ind_all]).ravel("F")
        elif neu_rob_ind.size > 0:
            neu_rob_dir_ind = neu_rob_ind_all
        elif dir_ind.size > 0:
            neu_rob_dir_ind = dir_ind_all
        elif num_bound == 0:  # all of them are empty
            neu_rob_dir_ind = neu_rob_ind
        else:
            raise ValueError("Boundary values should be either Dirichlet or " "Neumann")

        num_subfno = subcell_topology.num_subfno_unique

        # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
        # Map these to all half-face indices.
        bnd_2_all_hf = sps.coo_matrix(
            (np.ones(num_bound), (np.arange(num_bound), neu_rob_dir_ind)),
            shape=(num_bound, num_subfno),
        )
        mat = sps.coo_matrix(
            (data, (rows, cols)), shape=(num_flux + num_rob + num_pr, num_bound)
        )
        rhs_bound = mat * bnd_2_all_hf

        return rhs_bound.tocsr()

    def _bc_for_subgrid(
        self,
        bc: pp.BoundaryCondition,
        sub_sd: pp.Grid,
        face_map: np.ndarray,
        full_grid: pp.Grid,
    ) -> pp.BoundaryCondition:
        """Obtain a representation of a boundary condition for a subgrid of
        the original grid.

        This is somehow better fit for the BoundaryCondition class, but it is not clear
        whether the implementation is sufficiently general to be put there.

        Parameters:
            sub_sd (pp.Grid): Grid for which the new condition applies. Is
                assumed to be a subgrid of the grid to initialize this object.
            face_map (np.ndarray): Index of faces of the original grid from
                which the new conditions should be picked.

        Returns:
            BoundaryCondition: New bc object, aimed at a smaller grid.
            Will have type of boundary condition, basis and robin_weight copied
            from the specified faces in the original grid.

        """

        sub_bc = pp.BoundaryCondition(sub_sd)

        # In certain cases (EK: DFM upscaling) it may be of interest to set boundary
        # conditions for non-boundary faces - don't ask. If such conditions are found,
        # we copy them to the subgrid.
        neu_condition_on_interior = np.setdiff1d(
            np.where(bc.is_neu), full_grid.get_all_boundary_faces()
        )
        if neu_condition_on_interior.size > 0:
            hit = np.isin(face_map, neu_condition_on_interior)
            sub_bc.is_neu[hit] = True

        sub_bc.is_dir = bc.is_dir[face_map]
        sub_bc.is_rob = bc.is_rob[face_map]
        sub_bc.is_neu[sub_bc.is_dir + sub_bc.is_rob] = False

        sub_bc.robin_weight = bc.robin_weight[face_map]
        sub_bc.basis = bc.basis[face_map]

        return sub_bc

    def _constit_for_subgrid(
        self, K: pp.SecondOrderTensor, loc_cells: np.ndarray
    ) -> pp.SecondOrderTensor:
        """Extract the permeability tensor for a subgrid.

        Parameters:
            K: Permeability tensor for the full grid.
            loc_cells: Indices of the cells in the subgrid.

        Returns:
            Permeability tensor for the subgrid.

        """
        # Copy stiffness tensor, and restrict to local cells
        loc_K = K.copy()
        loc_K.values = loc_K.values[::, ::, loc_cells]
        return loc_K


def reconstruct_presssure(
    sd: pp.Grid, subcell_topology, eta: Optional[float | np.ndarray] = None
) -> tuple[sps.spmatrix, sps.spmatrix]:
    """Function for reconstructing the pressure at the half faces given the
    local gradients.

    For a subcell Ks associated with cell K and node s, the pressure at a point x is
    given by

        p_Ks + G_Ks (x - x_k),

    x_K is the cell center of cell k. The point at which we evaluate the pressure is
    given by eta, which is equivalent to the continuity points in mpfa. For an internal
    subface we will obtain two values for the pressure, one for each of the cells
    associated with the subface. The pressure given here is the average of the two. Note
    that at the continuity points the two pressures will by construction be equal.

    Parameters:
        sd: Grid
        subcell_topology: Wrapper class for numbering of subcell faces, cells
            etc.
        eta (range=[0,1)): Parameter determining the point
            at which the pressures is evaluated. If ``eta`` is a ``nd-array`` it should
            be on the size of subcell_topology.num_subfno. If ``eta`` is not given the
            method will call ``:meth:~pp.numerics.fv.fvutils.determine_eta`` to set it.

    Returns:
        :obj:`~scipy.sparse.spmatrix`: ``(shape=(num_half_faces, sd.num_cells))``

            Pressure reconstruction for the pressure at the half faces. This is the
            contribution from the cell-center pressures.
        :obj:`~scipy.sparse.spmatrix`: ``(shape=(num_half_faces, sd.num_cells))``

            Pressure reconstruction for the pressures at the half faces. This is the
            contribution from the boundary conditions.

    """

    if eta is None:
        eta = pp.fvutils.determine_eta(sd)

    # Calculate the distance from the cell centers to continuity points.
    D_g = pp.fvutils.compute_dist_face_cell(
        sd, subcell_topology, eta, return_paired=False
    )
    # We here average the contribution on internal sub-faces. To get out both pressures
    # on a sub-face, remove the averaging.
    _, IC, counts = np.unique(
        subcell_topology.subfno, return_inverse=True, return_counts=True
    )

    avg_over_subfaces = sps.coo_matrix(
        (1 / counts[IC], (subcell_topology.subfno, subcell_topology.subhfno))
    ).tocsr()
    D_g = avg_over_subfaces * D_g
    D_g = D_g.tocsr()

    # Get a mapping from cell centers to half-faces.
    D_c = sps.coo_matrix(
        (1 / counts[IC], (subcell_topology.subfno, subcell_topology.cno))
    ).tocsr()

    return D_g, D_c
