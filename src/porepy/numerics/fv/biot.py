""" Modules contains discretization of poro-elasticity by the multi-point stress
approximation.

The discretization scheme is described in

    J.M. Nordbotten (2016): STABLE CELL-CENTERED FINITE VOLUME DISCRETIZATION FOR BIOT
        EQUATIONS, SINUM.

The module contains four classes: Biot is the main class, responsible for discretization
of the poro-elastic system (relying heavily on its parent class pp.Mpsa). This is a
proper discretization class, in that it has discretize() and assemble_matrix_rhs()
methods.

The other classes, GradP, DivU and BiotStabilization, are containers for the other
terms in the poro-elasticity system. These have empty discretize() methods, and should
not be used directly.

Because of the number of variables and equations, and their somewhat difficult relation,
the most convenient way to set up a discretization for poro-elasticity is to use
:class:`~porepy.models.poromechanics.Poromechanics` (designed for fractures and contact
mechanics, but will turn into a standard poro-elasticity equation for non-fractured
domains).

"""

from __future__ import annotations

import logging
from time import time
from typing import Any, Literal

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization

# Module-wide logger
logger = logging.getLogger(__name__)


class Biot(pp.Mpsa):
    """Discretization class for poro-elasticity, based on MPSA.

    See also pp.Mpsa for further attributes.
    This discretization is an extension of the pure mechanical MPSA discretization, see
    `:class:~porepy.numerics.fv.mpsa.Mpsa` for documentation. In addition to the pure
    mechanical discretization, this class discretizes the coupling terms between
    mechanics and flow. Specifically, the following coupling terms are discretized:

    ``data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]["grad_p"]``:
    Discretization of the term :math:`\\nabla p` in the mechanics part of the
    poromechanical system.

    ``data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]["bound_displacement_pressure"]``:
    Discretization of the pressure contribution to the boundary displacement trace
    reconstruction in a poromechanical system, see class documentation in
    `:class:~porepy.numerics.fv.mpsa.Mpsa` for more details.

    ``data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]["div_u"]``: Discretization of
    the term :math:`\\nabla \\cdot u` in the mass balance part of the poromechanical
    system.

    ``data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]["bound_div_u"]``:
    Discretization of boundary conditions for the term :math:`\\nabla \\cdot u` in the
    mass balance part of the poromechanical system.

    ``data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]["biot_stabilization"]``:
    Numerical stabilization term (essentially extra pressure diffusion) needed to
    stabilize the pressure discretization, see Nordbotten 2016 for a derivation.

    NOTE: This class cannot be used for assembly. Use
    :class:`~porepy.models.poromechanics.Poromechanics` instead; there one can also find
    examples on how to use the individual discretization matrices.

    """

    def __init__(
        self,
        mechanics_keyword: str = "mechanics",
        flow_keyword: str = "flow",
        vector_variable: str = "displacement",
        scalar_variable: str = "pressure",
    ) -> None:
        """Set the two keywords.

        The keywords are used to access and store parameters and discretization
        matrices.
        """
        super().__init__("")
        self.mechanics_keyword = mechanics_keyword
        """Keyword used to identify the parameter dictionary associated with the
        mechanics subproblem."""
        self.flow_keyword = flow_keyword
        """Keyword used to identify the parameter dictionary associated with the
        flow subproblem."""
        # Set variable names for the vector and scalar variable, used to access
        # solutions from previous time steps
        self.vector_variable = vector_variable
        """Name for the vector variable, used throughout the discretization and solution."""
        self.scalar_variable = scalar_variable
        """Name for the scalar variable, used throughout the discretization and solution."""

        # Strings used to identify discretization matrices for various terms constructed
        # by this class. Hardcoded here to enforce a common standard.
        # Since these keys are used also for discretizations of other terms in Biot
        # (GradP, DivU), the keys should perhaps have been module level constants,
        # but this would have broken with the system in other discretizaitons.
        self.div_u_matrix_key = "div_u"
        """Keyword used to identify the discretization matrix of the term div(u)."""
        self.bound_div_u_matrix_key = "bound_div_u"
        """Keyword used to identify the discretization matrix of the boundary condition
        for the term div(u)."""
        self.grad_p_matrix_key = "grad_p"
        """Keyword used to identify the discretization matrix of the term grad(p)."""
        self.stabilization_matrix_key = "biot_stabilization"
        """Keyword used to identify the discretization matrix for the stabilization
        term."""
        self.bound_pressure_matrix_key = "bound_displacement_pressure"
        """Keyword used to identify the discretization matrix for the pressure
        contribution to boundary displacement reconstruction."""

        self.mass_matrix_key = "mass"

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.

        Parameters:
            sd: A grid.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_cells * (1 + sd.dim)

    def assemble_matrix_rhs(
        self, sd: pp.Grid, sd_data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Method inherited from superclass should not be used.

        Parameters:
            sd: Grid to be discretized.
            sd_data: Data dictionary for this grid.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The size of
                the matrix will depend on the specific discretization.
            np.ndarray: Right-hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.

        Raises:
            NotImplementedError: If invoked. The method is included to be compatible
                with the wider discretization class, but assembly should be handled
                by :class:`~porepy.models.poromechanics.Poromechanics`.

        """
        raise NotImplementedError(
            """This class cannot be used for assembly.
    Use :class:`~porepy.models.poromechanics.Poromechanics` instead."""
        )

    def update_discretization(self, sd: pp.Grid, sd_data: dict):
        """Update discretization.

        The updates can generally come as a combination of two forms:
            1) The discretization on part of the grid should be recomputed.
            2) The old discretization can be used (in parts of the grid), but the
               numbering of unknowns has changed, and the discretization should be
               reordered accordingly.

        Information on the basis for the update should be stored in a field

            sd_data['update_discretization']

        This should be a dictionary which could contain keys:

            modified_cells, modified_faces


        define cells, faces and nodes that have been modified (either parameters,
        geometry or topology), and should be rediscretized. It is up to the
        discretization method to implement the change necessary by this modification.
        Note that depending on the computational stencil of the discretization method,
        a grid quantity may be rediscretized even if it is not marked as modified.

        The dictionary sd_data['update_discretization'] should further have keys:

            cell_index_map, face_index_map

        these should specify sparse matrices that map old to new indices. If not
        provided, the cell and face bookkeeping will be assumed constant.

        Parameters:
            sd: Grid to be rediscretized.
            sd_data: With discretization parameters.

        """

        # If neither cells nor faces have been modified, we should not start the update
        # procedure (it will result in a key error towards the end of this function - it
        # is technical). If no updates, we shortcut the method
        update_info = sd_data["update_discretization"]
        # By default, neither cells nor faces have been updated
        update_cells = update_info.get("modified_cells", np.array([], dtype=int))
        update_faces = update_info.get("modified_faces", np.array([], dtype=int))

        if update_cells.size == 0 and update_faces.size == 0:
            return

        # The implementation is quite a bit more involved than the corresponding methods
        # for mpfa and mpsa, due to the multi-physics structure of the discretization.

        # Matrices computed by self._discretized_mech, but associtated with self.flow_keyword.
        # These are moved to the self.mechanics_keyword matrix dictionary, and will then be
        # moved back again at the end of this function
        mech_in_flow = [
            self.div_u_matrix_key,
            self.bound_div_u_matrix_key,
            self.stabilization_matrix_key,
        ]
        for key in mech_in_flow:
            mat = sd_data[pp.DISCRETIZATION_MATRICES][self.flow_keyword].pop(key)
            sd_data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword][key] = mat

        # Dump discretization of mass term - this will be fully rediscretized below
        sd_data[pp.DISCRETIZATION_MATRICES][self.flow_keyword].pop(
            self.mass_matrix_key, None
        )

        # Define which of the matrices should be considered cell and face quantities,
        # vector and scalar.
        scalar_cell_right = [
            self.grad_p_matrix_key,
            self.stabilization_matrix_key,
            self.bound_pressure_matrix_key,
        ]
        vector_cell_right = [
            self.stress_matrix_key,
            self.bound_displacement_cell_matrix_key,
            self.div_u_matrix_key,
        ]
        vector_face_right = [
            self.bound_stress_matrix_key,
            self.bound_displacement_face_matrix_key,
            self.bound_div_u_matrix_key,
        ]

        scalar_cell_left = [
            self.div_u_matrix_key,
            self.stabilization_matrix_key,
            self.bound_div_u_matrix_key,
        ]

        vector_face_left = [
            self.stress_matrix_key,
            self.grad_p_matrix_key,
            self.bound_stress_matrix_key,
            self.bound_displacement_cell_matrix_key,
            self.bound_displacement_face_matrix_key,
            self.bound_pressure_matrix_key,
        ]

        # Update discretization. As part of the process, the mech_in_flow matrices
        # are moved back to the flow matrix dictionary.
        pp.fvutils.partial_update_discretization(
            sd,
            sd_data,
            self.mechanics_keyword,
            self.discretize,
            dim=sd.dim,
            scalar_cell_right=scalar_cell_right,
            vector_cell_right=vector_cell_right,
            vector_face_right=vector_face_right,
            vector_face_left=vector_face_left,
            scalar_cell_left=scalar_cell_left,
            second_keyword=self.flow_keyword,
        )
        # Remove the mech_in_flow matrices from the mechanics dictionary
        for key in mech_in_flow:
            sd_data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword].pop(key, None)

    def discretize(self, sd: pp.Grid, sd_data: dict) -> None:
        """Discretize the mechanics terms in a poromechanical system.

        NOTE: This function does *not* discretize purely flow-related terms (Darcy flow
        and compressibility).

        The parameters needed for the discretization are stored in the dictionary
        sd_data, which should contain the following mandatory keywords:

            Related to mechanics equation (in
            sd_data[pp.PARAMETERS][self.mechanics_keyword]):
                fourth_order_tensor: Fourth order tensor representing elastic moduli.
                bc: BoundaryCondition object for mechanics equation.
                    Used in mpsa.

        In addition, the following parameters are optional:
            Related to coupling terms:
                biot_alpha (double between 0 and 1): Biot's coefficient.
                    Defaults to 1.

            Related to numerics:
                - inverter (``str``): Which method to use for block inversion. See
                    pp.fvutils.invert_diagonal_blocks for detail, and for default
                    options.
                - mpfa_eta (``float``): Location of continuity point in MPSA. Defaults
                    to 1/3 for simplex grids, 0 otherwise.
                - partition_arguments (``dict``): Arguments to control the number
                    of subproblems used to discretize the grid. Can be either the target
                    maximal memory use (controlled by keyword 'max_memory' in
                    ``partition_arguments``), or the number of subproblems (keyword
                    'num_subproblems' in ``partition_arguments``). If none are given,
                    the default is to use 1e9 bytes of memory per subproblem. If both
                    are given, the maximal memory use is prioritized.

        The discretization is stored in the data dictionary, in the form of several
        matrices representing different coupling terms. For details, and how to combine
        these, see self.assemble_matrix()

        Parameters:
            sd: Grid to be discretized.
            sd_data: Containing data for discretization. See above for specification.

        """
        parameter_dictionary: dict[str, Any] = sd_data[pp.PARAMETERS][
            self.mechanics_keyword
        ]
        matrices_m: dict[str, sps.spmatrix] = sd_data[pp.DISCRETIZATION_MATRICES][
            self.mechanics_keyword
        ]
        matrices_f: dict[str, sps.spmatrix] = sd_data[pp.DISCRETIZATION_MATRICES][
            self.flow_keyword
        ]
        bound: pp.BoundaryConditionVectorial = parameter_dictionary["bc"]
        constit: pp.FourthOrderTensor = parameter_dictionary["fourth_order_tensor"]

        eta: float = parameter_dictionary.get("mpsa_eta", pp.fvutils.determine_eta(sd))
        inverter: Literal["python", "numba"] = parameter_dictionary.get(
            "inverter", "numba"
        )

        alpha: float = parameter_dictionary["biot_alpha"]

        # Control of the number of subdomanis.
        max_memory, num_subproblems = pp.fvutils.parse_partition_arguments(
            parameter_dictionary.get("partition_arguments", {})
        )

        # Whether to update an existing discretization, or construct a new one.
        # If True, either specified_cells, _faces or _nodes should also be given, or
        # else a full new discretization will be computed
        update: bool = parameter_dictionary.get("update_discretization", False)

        # The discretization can be limited to a specified set of cells, faces or nodes
        # If none of these are specified, the entire grid will be discretized.
        # NOTE: active_faces are all faces to have their stencils updated, while
        # active_cells may form a larger set (to accurately update all faces on a
        # subgrid, it is necessary to assign some overlap in terms cells).
        active_cells, active_faces = pp.fvutils.find_active_indices(
            parameter_dictionary, sd
        )

        # Extract a grid, and get global indices of its active faces and nodes
        active_grid, extracted_faces, extracted_nodes = pp.partition.extract_subgrid(
            sd, active_cells
        )
        # Constitutive law and boundary condition for the active grid
        active_constit: pp.FourthOrderTensor = self._constit_for_subgrid(
            constit, active_cells
        )

        # Extract the relevant part of the boundary condition
        active_bound: pp.BoundaryConditionVectorial = self._bc_for_subgrid(
            bound, active_grid, extracted_faces
        )

        # Initialize matrices to store discretization
        nd = active_grid.dim
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
        peak_memory_estimate = self._estimate_peak_memory_mpsa(active_grid)

        # Holders for discretizations. There are quite a few items to keep track of, but
        # then the discretization does quite a few different things
        active_stress = sps.csr_matrix((nf * nd, nc * nd))
        active_bound_stress = sps.csr_matrix((nf * nd, nf * nd))
        active_grad_p = sps.csr_matrix((nf * nd, nc))
        active_div_u = sps.csr_matrix((nc, nc * nd))
        active_bound_div_u = sps.csr_matrix((nc, nf * nd))
        active_stabilization = sps.csr_matrix((nc, nc))
        active_bound_displacement_cell = sps.csr_matrix((nf * nd, nc * nd))
        active_bound_displacement_face = sps.csr_matrix((nf * nd, nf * nd))
        active_bound_displacement_pressure = sps.csr_matrix((nf * nd, nc))

        # Loop over all partition regions, construct local problems, and transfer
        # discretization to the entire active grid
        for reg_i, (
            sub_sd,
            faces_in_subgrid,
            cells_in_subgrid,
            l2g_cells,
            l2g_faces,
        ) in enumerate(
            pp.fvutils.subproblems(
                active_grid,
                peak_memory_estimate,
                max_memory=max_memory,
                num_subproblems=num_subproblems,
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

            tic = time()
            # Copy stiffness tensor, and restrict to local cells
            loc_c: pp.FourthOrderTensor = self._constit_for_subgrid(
                active_constit, l2g_cells
            )

            # Boundary conditions are slightly more complex. Find local faces
            # that are on the global boundary.
            # Then transfer boundary condition on those faces.
            loc_bnd: pp.BoundaryConditionVectorial = self._bc_for_subgrid(
                active_bound, sub_sd, l2g_faces
            )

            # Discretization of sub-problem
            (
                loc_stress,
                loc_bound_stress,
                loc_div_u,
                loc_bound_div_u,
                loc_grad_p,
                loc_biot_stab,
                loc_bound_displacement_cell,
                loc_bound_displacement_face,
                loc_bound_displacement_pressure,
            ) = self._local_discretization(
                sub_sd, loc_c, loc_bnd, alpha, eta=eta, inverter=inverter
            )

            # Eliminate contribution from faces already discretized (the dual grids /
            # interaction regions may be structured so that some faces have previously
            # been partially discretized even if it has not been their turn until now)
            eliminate_face = np.where(
                np.logical_not(np.in1d(l2g_faces, faces_in_subgrid))
            )[0]
            pp.fvutils.remove_nonlocal_contribution(
                eliminate_face,
                sd.dim,
                loc_stress,
                loc_bound_stress,
                loc_bound_displacement_cell,
                loc_bound_displacement_face,
                loc_grad_p,
                loc_bound_displacement_pressure,
            )

            eliminate_cell = np.where(
                np.logical_not(np.in1d(l2g_cells, cells_in_subgrid))
            )[0]
            pp.fvutils.remove_nonlocal_contribution(
                eliminate_cell, 1, loc_div_u, loc_bound_div_u, loc_biot_stab
            )

            # Next, transfer discretization matrices from the local to the active grid
            # Get a mapping from the local to the active grid
            face_map_vec, cell_map_vec = pp.fvutils.map_subgrid_to_grid(
                active_grid, l2g_faces, l2g_cells, is_vector=True
            )
            face_map_scalar, cell_map_scalar = pp.fvutils.map_subgrid_to_grid(
                active_grid, l2g_faces, l2g_cells, is_vector=False
            )

            # Update discretization on the active grid.
            active_stress += face_map_vec * loc_stress * cell_map_vec
            active_bound_stress += (
                face_map_vec * loc_bound_stress * face_map_vec.transpose()
            )

            # Update global face fields.
            active_bound_displacement_cell += (
                face_map_vec * loc_bound_displacement_cell * cell_map_vec
            )
            active_bound_displacement_face += (
                face_map_vec * loc_bound_displacement_face * face_map_vec.transpose()
            )

            active_stabilization += (
                cell_map_scalar.transpose() * loc_biot_stab * cell_map_scalar
            )

            active_grad_p += face_map_vec * loc_grad_p * cell_map_scalar
            active_div_u += cell_map_scalar.transpose() * loc_div_u * cell_map_vec
            active_bound_div_u += (
                cell_map_scalar.transpose() * loc_bound_div_u * face_map_vec.transpose()
            )

            active_bound_displacement_pressure += (
                face_map_vec * loc_bound_displacement_pressure * cell_map_scalar
            )
            logger.info(f"Done with subproblem {reg_i}. Elapsed time {time() - tic}")
            # Done with this subdomain, move on to the next one

        # Divide by the number of times a face has been discretized. This is necessary
        # to avoid double counting of faces on the boundary between subproblems. Note
        # that this is done before mapping from the active to the full grid, since the
        # subgrids (thus face map) was computed on the active grid.
        #
        # IMPLEMENTATION NOTE: With the current implementation, this scaling is only
        # needed for the vector quantities, that is, not for the stabilization and div_u
        # terms. This is because the latter are computed cell-wise rather than
        # face-wise, and it so happens that this is sufficient to avoid double counting.
        # If we ever change the implementation (e.g., when introducing tensor Biot
        # coefficients), it is likely the scaling will be needed also for the scalar
        # terms.
        num_face_repetitions_vector = np.tile(
            np.bincount(np.concatenate(faces_in_subgrid_accum)), (nd, 1)
        ).ravel("F")
        scaling_vector = sps.dia_matrix(
            (1.0 / num_face_repetitions_vector, 0), shape=(nf * nd, nf * nd)
        )

        # Vector fields
        active_stress = scaling_vector @ active_stress
        active_bound_stress = scaling_vector @ active_bound_stress
        active_bound_displacement_cell = scaling_vector @ active_bound_displacement_cell
        active_bound_displacement_face = scaling_vector @ active_bound_displacement_face
        active_grad_p = scaling_vector @ active_grad_p
        active_bound_displacement_pressure = (
            scaling_vector @ active_bound_displacement_pressure
        )

        # We are done with the discretization. What remains is to map the computed
        # matrices back from the active grid to the full one.
        face_map_vec, cell_map_vec = pp.fvutils.map_subgrid_to_grid(
            sd, extracted_faces, active_cells, is_vector=True
        )
        face_map_scalar, cell_map_scalar = pp.fvutils.map_subgrid_to_grid(
            sd, extracted_faces, active_cells, is_vector=False
        )

        stress = face_map_vec * active_stress * cell_map_vec
        bound_stress = face_map_vec * active_bound_stress * face_map_vec.transpose()

        # Update global face fields.
        bound_displacement_cell = (
            face_map_vec * active_bound_displacement_cell * cell_map_vec
        )
        bound_displacement_face = (
            face_map_vec * active_bound_displacement_face * face_map_vec.transpose()
        )

        stabilization = (
            cell_map_scalar.transpose() * active_stabilization * cell_map_scalar
        ).tocsr()

        grad_p = face_map_vec * active_grad_p * cell_map_scalar
        div_u = (cell_map_scalar.transpose() * active_div_u * cell_map_vec).tocsr()
        bound_div_u = (
            cell_map_scalar.transpose() * active_bound_div_u * face_map_vec.transpose()
        ).tocsr()

        bound_displacement_pressure = (
            face_map_vec * active_bound_displacement_pressure * cell_map_scalar
        )

        # Eliminate any contributions not associated with the active grid
        eliminate_faces = np.setdiff1d(np.arange(sd.num_faces), active_faces)
        pp.fvutils.remove_nonlocal_contribution(
            eliminate_faces,
            sd.dim,
            stress,
            bound_stress,
            bound_displacement_cell,
            bound_displacement_face,
            grad_p,
            bound_displacement_pressure,
        )

        # Cells to be updated is a bit more involved. Best guess now is to update
        # all cells that have had one of its faces updated. This may not be correct for
        # general combinations of specified cells, nodes and faces.
        tmp = sd.cell_faces.transpose()
        tmp.data = np.abs(tmp.data)
        af_vec = np.zeros(sd.num_faces, dtype=bool)
        af_vec[active_faces] = 1
        update_cell_ind = np.where(tmp * af_vec)[0]
        eliminate_cells = np.setdiff1d(np.arange(sd.num_cells), update_cell_ind)
        pp.fvutils.remove_nonlocal_contribution(
            eliminate_cells, 1, div_u, bound_div_u, stabilization
        )

        # Either update the discretization scheme, or store the full one
        if update:
            # The faces to be updated are given by active_faces
            update_face_ind = pp.fvutils.expand_indices_nd(active_faces, sd.dim)

            matrices_m[self.stress_matrix_key][update_face_ind] = stress[
                update_face_ind
            ]
            matrices_m[self.bound_stress_matrix_key][update_face_ind] = bound_stress[
                update_face_ind
            ]
            matrices_f[self.div_u_matrix_key][update_cell_ind] = div_u[update_cell_ind]
            matrices_f[self.bound_div_u_matrix_key][update_cell_ind] = bound_div_u[
                update_cell_ind
            ]
            matrices_m[self.grad_p_matrix_key][update_face_ind] = grad_p[
                update_face_ind
            ]
            matrices_f[self.stabilization_matrix_key][update_cell_ind] = stabilization[
                update_cell_ind
            ]
            matrices_m[self.bound_displacement_cell_matrix_key][update_face_ind] = (
                bound_displacement_cell[update_face_ind]
            )
            matrices_m[self.bound_displacement_face_matrix_key][update_face_ind] = (
                bound_displacement_face[update_face_ind]
            )
            matrices_m[self.bound_pressure_matrix_key][update_face_ind] = (
                bound_displacement_pressure[update_face_ind]
            )
        else:
            matrices_m[self.stress_matrix_key] = stress
            matrices_m[self.bound_stress_matrix_key] = bound_stress
            matrices_m[self.grad_p_matrix_key] = grad_p
            matrices_m[self.bound_displacement_cell_matrix_key] = (
                bound_displacement_cell
            )
            matrices_m[self.bound_displacement_face_matrix_key] = (
                bound_displacement_face
            )
            matrices_m[self.bound_pressure_matrix_key] = bound_displacement_pressure

            matrices_f[self.div_u_matrix_key] = div_u
            matrices_f[self.bound_div_u_matrix_key] = bound_div_u
            matrices_f[self.stabilization_matrix_key] = stabilization

    def _local_discretization(
        self,
        sd: pp.Grid,
        constit: pp.FourthOrderTensor,
        bound_mech: pp.BoundaryConditionVectorial,
        alpha: float,
        eta: float,
        inverter: Literal["python", "numba"],
        hf_output: bool = False,
    ) -> tuple[
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
        sps.spmatrix,
    ]:
        """Discretization of poro-elastic system."""

        # The grid coordinates are always three-dimensional, even if the grid
        # is really 2D. This means that there is not a 1-1 relation between the
        # number of coordinates of a point / vector and the real dimension.
        # This again violates some assumptions tacitly made in the
        # discretization (in particular that the number of faces of a cell that
        # meets in a vertex equals the grid dimension, and that this can be
        # used to construct an index of local variables in the discretization).
        # These issues should be possible to overcome, but for the moment, we
        # simply force 2D grids to be proper 2D.
        if sd.dim == 2:
            sd, constit = self._reduce_grid_constit_2d(sd, constit)

        nd = sd.dim

        # Define subcell topology
        subcell_topology = pp.fvutils.SubcellTopology(sd)
        # The boundary conditions must be given on the subfaces
        if bound_mech.num_faces == subcell_topology.num_subfno_unique:
            subface_rhs = True
        else:
            # If they are given on the faces, expand the boundary conditions
            bound_mech = pp.fvutils.boundary_to_sub_boundary(
                bound_mech, subcell_topology
            )
            subface_rhs = False

        # Obtain mappings to exclude boundary faces for mechanics
        bound_exclusion_mech = pp.fvutils.ExcludeBoundaries(
            subcell_topology, bound_mech, nd
        )

        # Call core part of MPSA
        hook, igrad, cell_node_blocks = self._create_inverse_gradient_matrix(
            sd, constit, subcell_topology, bound_exclusion_mech, eta, inverter
        )
        num_sub_cells = cell_node_blocks.shape[0]
        # Right-hand side terms for the stress discretization
        rhs_cells: sps.spmatrix = self._create_rhs_cell_center(
            sd, subcell_topology, eta, num_sub_cells, bound_exclusion_mech
        )

        # Stress discretization
        stress = hook * igrad * rhs_cells

        # Right-hand side for boundary discretization
        rhs_bound = self._create_bound_rhs(
            bound_mech, bound_exclusion_mech, subcell_topology, sd, subface_rhs
        )
        # Discretization of boundary values
        bound_stress = hook * igrad * rhs_bound

        if not hf_output:
            # If the boundary condition is given for faces we return the discretization
            # on for the face values. Otherwise, it is defined for the subfaces.
            hf2f = pp.fvutils.map_hf_2_f(
                subcell_topology.fno_unique, subcell_topology.subfno_unique, nd
            )
            bound_stress = hf2f * bound_stress * hf2f.T
            stress = hf2f * stress
            rhs_bound = rhs_bound * hf2f.T

        # trace of strain matrix
        div = self._subcell_gradient_to_cell_scalar(sd, cell_node_blocks)
        div_u = div * igrad * rhs_cells

        # The boundary discretization of the div_u term is represented directly
        # on the cells, instead of going via the faces.
        bound_div_u = div * igrad * rhs_bound

        # Call discretization of grad_p-term
        rhs_jumps, grad_p_face = self._create_rhs_grad_p(
            sd, subcell_topology, alpha, bound_exclusion_mech
        )

        if hf_output:
            # If boundary conditions are given on subfaces we keep the subface
            # discretization
            grad_p = hook * igrad * rhs_jumps + grad_p_face
        else:
            # otherwise, we map it to faces
            grad_p = hf2f * (hook * igrad * rhs_jumps + grad_p_face)

        # consistency term for the flow equation
        stabilization = div * igrad * rhs_jumps

        # We obtain the reconstruction of displacements. This is equivalent as for
        # mpsa, but we get a contribution from the pressures.
        dist_grad, cell_centers = self._reconstruct_displacement(
            sd, subcell_topology, eta
        )

        disp_cell = dist_grad * igrad * rhs_cells + cell_centers
        disp_bound = dist_grad * igrad * rhs_bound
        disp_pressure = dist_grad * igrad * rhs_jumps

        if not hf_output:
            # hf2f sums the values, but here we need an average.
            # For now, use simple average, although area weighted values may be more accurate
            num_subfaces = hf2f.sum(axis=1).A.ravel()
            scaling = sps.dia_matrix(
                (1.0 / num_subfaces, 0), shape=(hf2f.shape[0], hf2f.shape[0])
            )

            disp_cell = scaling * hf2f * disp_cell
            disp_bound = scaling * hf2f * disp_bound
            disp_pressure = scaling * hf2f * disp_pressure

        return (
            stress,
            bound_stress,
            div_u,
            bound_div_u,
            grad_p,
            stabilization,
            disp_cell,
            disp_bound,
            disp_pressure,
        )

        # Add discretizations to data

    def _create_rhs_grad_p(
        self,
        sd: pp.Grid,
        subcell_topology: pp.fvutils.SubcellTopology,
        alpha: float,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
    ) -> tuple[sps.spmatrix, sps.spmatrix]:
        """
        Consistent discretization of grad_p-term in MPSA-W method.

        Parameters:
            sd: grid to be discretized.
            subcell_topology: Wrapper class for numbering of subcell faces, cells
                etc.
            alpha: Biot's coupling coefficient, given as a scalar in input
            bound_exclusion: Object that can eliminate faces related to boundary
                conditions.

        Returns:
            scipy.sparse.csr_matrix (shape num_subcells * dim, num_cells):
            discretization of the jumps in [n alpha p] term,
            ready to be multiplied with inverse gradient
            scipy.sparse.csr_matrix (shape num_subfaces * dim, num_cells):
                discretization of the force on the face due to cell-centre
                pressure from a unique side.

        """
        """
        Method properties and implementation details.
        Basis functions, namely 'stress' and 'bound_stress', for the displacement
        discretization are obtained as in standard MPSA-W method.
        Pressure is represented as forces in the cells.
        However, jumps in pressure forces over a cell face act as force
        imbalance, and thus induce additional displacement gradients in the sub-cells.
        An additional system is set up, which applies non-zero conditions to the
        traction continuity equation. This can be expressed as a linear system on the form

            (i)   A * grad_u            = I
            (ii)  B * grad_u + C * u_cc = 0
            (iii) 0            D * u_cc = 0

        Thus (i)-(iii) can be inverted to express the additional displacement gradients
        due to imbalance in pressure forces as in terms of the cell center variables.
        Thus we can compute the basis functions 'grad_p_jumps' on the sub-cells.
        To ensure traction continuity, as soon as a convention is chosen for what side
        the force evaluation should be considered on, an additional term, called
        'grad_p_face', is added to the full force. This latter term represents the force
        due to cell-center pressure acting on the face from the chosen side.
        The pair subfno_unique-unique_subfno gives the side convention.
        The full force on the face is therefore given by

        t = stress * u + bound_stress * u_b + alpha * (grad_p_jumps + grad_p_face) * p

        The strategy is as follows.
        1. compute product normal_vector * alpha and get a map for vector problems
        2. assemble r.h.s. for the new linear system, needed for the term 'grad_p_jumps'
        3. compute term 'grad_p_face'
        """

        nd = sd.dim

        num_subhfno = subcell_topology.subhfno.size
        num_subfno_unique = subcell_topology.num_subfno_unique
        num_subfno = subcell_topology.num_subfno

        # Step 1

        # The implementation is valid for tensor Biot coefficients, but for the
        # moment, we only allow for scalar inputs.
        # Take Biot's alpha as a tensor
        alpha_tensor = pp.SecondOrderTensor(alpha * np.ones(sd.num_cells))

        if nd == 2:
            alpha_tensor.values = np.delete(alpha_tensor.values, (2), axis=0)
            alpha_tensor.values = np.delete(alpha_tensor.values, (2), axis=1)

        # Obtain normal_vector * alpha, pairings of cells and nodes (which together
        # uniquely define sub-cells, and thus index for gradients)
        (
            nAlpha_grad,
            cell_node_blocks,
            sub_cell_index,
        ) = pp.fvutils.scalar_tensor_vector_prod(sd, alpha_tensor, subcell_topology)
        # transfer nAlpha to a subface-based quantity by pairing expressions on the
        # two sides of the subface
        unique_nAlpha_grad = subcell_topology.pair_over_subfaces(nAlpha_grad)

        # convenience method for reshaping nAlpha from face-based
        # to component-based. This is to build a block diagonal sparse matrix
        # compatible with igrad * rhs_units, that is first all x-component, then y, and z

        def map_tensor(mat, nd, ind):
            newmat = mat[:, ind[0]]

            for i in range(1, nd):
                this_dim = mat[:, ind[i]]
                newmat = sps.block_diag([newmat, this_dim])

            return newmat

        # Reshape nAlpha component-wise
        nAlpha_grad = map_tensor(nAlpha_grad, nd, sub_cell_index)
        unique_nAlpha_grad = map_tensor(unique_nAlpha_grad, nd, sub_cell_index)

        # Step 2

        # The pressure term in the traction continuity equation is discretized
        # as a force on the faces. The right-hand side is thus formed of the
        # unit vector.

        def build_rhs_units_single_dimension(dim):
            # EK: Can we skip argument dim?
            vals = np.ones(num_subfno_unique)
            ind = subcell_topology.subfno_unique
            mat = sps.coo_matrix(
                (vals, (ind, ind)), shape=(num_subfno_unique, num_subfno_unique)
            )
            return mat

        rhs_units = build_rhs_units_single_dimension(0)

        for i in range(1, nd):
            this_dim = build_rhs_units_single_dimension(i)
            rhs_units = sps.block_diag([rhs_units, this_dim])

        # We get the sign of the subfaces. This will be needed for the boundary
        # faces if the normal vector points inn. This is because boundary
        # conditions always are set as if the normals point out.
        sgn = sd.cell_faces[
            subcell_topology.fno_unique, subcell_topology.cno_unique
        ].A.ravel("F")
        # NOTE: For some reason one should not multiply with the sign, but I don't
        # understand why. It should not matter much for the Biot alpha term since
        # by construction the biot_alpha_jumps and biot_alpha_force will cancel for
        # Neumann boundaries. We keep the sign matrix as an Identity matrix to remember
        # where it should be multiplied:
        sgn_nd = np.tile(np.abs(sgn), (sd.dim, 1))

        # In the local systems the coordinates are C ordered (first all x, then all y,
        # etc.), while they are ordered as F (first x,y,z of subface 1 then x,y,z of
        # subface 2) elsewhere. If there is a problem with the stabilization or the
        # boundary, this might be the place to start debugging. The fno_unique
        # and cno_unique is chosen such that the face normal of fno points out of cno
        # for internal faces, thus, sgn_diag_F/C will only flip the sign at the boundary.
        sgn_diag_F = sps.diags(sgn_nd.ravel("F"))
        sgn_diag_C = sps.diags(sgn_nd.ravel("C"))

        # Recall the ordering of the local equations:
        # First stress equilibrium for the internal subfaces.
        # Then the stress equilibrium for the Neumann subfaces.
        # Then the Robin subfaces.
        # And last, the displacement continuity on both internal and external subfaces.
        rhs_int = bound_exclusion.exclude_boundary(rhs_units)
        rhs_neu = bound_exclusion.keep_neumann(sgn_diag_C * rhs_units)
        rhs_rob = bound_exclusion.keep_robin(sgn_diag_C * rhs_units)

        num_dir_subface = (
            bound_exclusion.exclude_neu_rob.shape[1]
            - bound_exclusion.exclude_neu_rob.shape[0]
        )

        # No right-hand side for cell displacement equations.
        rhs_units_displ_var = sps.coo_matrix(
            (nd * num_subfno - num_dir_subface, num_subfno_unique * nd)
        )

        # We get a plus because the -n * I * alpha * p term is moved over to the rhs
        # in the local systems
        rhs_units = sps.vstack([rhs_int, rhs_neu, rhs_rob, rhs_units_displ_var])

        del rhs_units_displ_var

        # Output should be on cell-level (not sub-cell)
        sc2c = pp.fvutils.cell_scalar_to_subcell_vector(
            sd.dim, sub_cell_index, cell_node_blocks[0]
        )

        # prepare for computation of imbalance coefficients,
        # that is jumps in cell-centers pressures, ready to be
        # multiplied with inverse gradients
        rhs_jumps = rhs_units * unique_nAlpha_grad * sc2c

        # Step 3

        # mapping from subface to unique subface for vector problems.
        # This mapping gives the convention from which side
        # the force should be evaluated on.
        vals = np.ones(num_subfno_unique * nd)
        rows = pp.fvutils.expand_indices_nd(subcell_topology.subfno_unique, nd)
        cols = pp.fvutils.expand_indices_incr(
            subcell_topology.unique_subfno, nd, num_subhfno
        )
        map_unique_subfno = sps.coo_matrix(
            (vals, (rows, cols)), shape=(num_subfno_unique * nd, num_subhfno * nd)
        )

        del vals, rows, cols

        # Prepare for computation of -grad_p_face term
        # Note that sgn_diag_F might only flip the boundary signs. See comment above.
        grad_p_face = -sgn_diag_F * map_unique_subfno * nAlpha_grad * sc2c

        return rhs_jumps, grad_p_face

    def _face_vector_to_scalar(self, nf: int, nd: int) -> sps.coo_matrix:
        """Create a mapping from vector quantities on faces (stresses) to scalar
        quantities. The mapping is intended for the boundary discretization of the
        displacement divergence term  (coupling term in the flow equation).

        Parameters:
            nf (int): Number of faces in the grid
        """
        rows = np.tile(np.arange(nf), ((nd, 1))).reshape((1, nd * nf), order="F")[0]

        cols = pp.fvutils.expand_indices_nd(np.arange(nf), nd)
        vals = np.ones(nf * nd)
        return sps.coo_matrix((vals, (rows, cols))).tocsr()

    def _subcell_gradient_to_cell_scalar(
        self, sd: pp.Grid, cell_node_blocks: np.ndarray
    ) -> sps.spmatrix:
        """Create a mapping from sub-cell gradients to cell-wise traces of the gradient
        operator. The mapping is intended for the discretization of the term div(u)
        (coupling term in flow equation).
        """
        # To pick out the trace of the strain tensor, we access elements
        #   (2d): 0 (u_x) and 3 (u_y)
        #   (3d): 0 (u_x), 4 (u_y), 8 (u_z)
        nd = sd.dim
        if nd == 2:
            trace = np.array([0, 3])
        elif nd == 3:
            trace = np.array([0, 4, 8])

        # Sub-cell wise trace of strain tensor: One row per sub-cell
        row, col = np.meshgrid(np.arange(cell_node_blocks.shape[1]), trace)
        # Adjust the columns to hit each sub-cell
        incr = np.cumsum(nd**2 * np.ones(cell_node_blocks.shape[1])) - nd**2
        col += incr.astype("int32")

        # Integrate the trace over the sub-cell, that is, distribute the cell
        # volumes equally over the sub-cells
        num_cell_nodes = sd.num_cell_nodes()
        cell_vol = sd.cell_volumes / num_cell_nodes
        val = np.tile(cell_vol[cell_node_blocks[0]], (nd, 1))
        # and we have our mapping from vector to scalar values on sub-cells
        vector_2_scalar = sps.coo_matrix(
            (val.ravel("F"), (row.ravel("F"), col.ravel("F")))
        ).tocsr()

        # Mapping from sub-cells to cells
        div_op = sps.coo_matrix(
            (
                np.ones(cell_node_blocks.shape[1]),
                (cell_node_blocks[0], np.arange(cell_node_blocks.shape[1])),
            )
        ).tocsr()
        # and the composed map
        div = div_op * vector_2_scalar
        return div

    # ----------------------- Methods for post processing -------------------------

    def compute_flux(self, sd: pp.Grid, u: np.ndarray, sd_data: dict) -> np.ndarray:
        """Compute flux field corresponding to a solution.

        Parameters:
            sd: Grid, or a subclass.
            u: Solution variable, representing displacements and
                pressure.
            sd_data: Dictionary related to grid and problem. Should
                contain boundary discretization.

        Returns:
            np.ndarray: Fluxes over all faces.

        """
        flux_discr = sd_data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]["flux"]
        bound_flux = sd_data[pp.DISCRETIZATION_MATRICES][self.flow_keyword][
            "bound_flux"
        ]
        bound_val = sd_data[pp.PARAMETERS][self.flow_keyword]["bc_values"]
        p = u[sd.dim * sd.num_cells :]
        flux = flux_discr * p + bound_flux * bound_val
        return flux

    def compute_stress(self, sd: pp.Grid, u: np.ndarray, sd_data: dict) -> np.ndarray:
        """Compute stress field corresponding to a solution.

        Parameters:
            sd: grid, or a subclass.
            u: Solution variable, representing displacements and pressure.
            sd_data: dictionary related to grid and problem. Should contain boundary
                discretization.

        Returns:
            np.ndarray, sd.dim * sd.num_faces: Stress over all faces. Stored as
                all stress values on the first face, then the second etc.

        """
        matrix_dictionary = sd_data[pp.DISCRETIZATION_MATRICES][self.mechanics_keyword]
        stress_discr = matrix_dictionary["stress"]
        bound_stress = matrix_dictionary["bound_stress"]
        bound_val = sd_data[pp.PARAMETERS][self.mechanics_keyword]["bc_values"]
        d = u[: sd.dim * sd.num_cells]

        stress = stress_discr * d + (bound_stress * bound_val)
        return stress


class GradP(Discretization):
    """Class for the pressure gradient term of the Biot equation."""

    def __init__(self, keyword: str) -> None:
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Parameters:
            keyword: Identifier of all information used for this discretization.

        """
        self.keyword = keyword
        # Use the same key to access the discretization matrix as the Biot class.
        self.grad_p_matrix_key = Biot().grad_p_matrix_key

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.

        Parameters:
            sd: A grid.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.dim * sd.num_cells

    def discretize(self, sd: pp.Grid, sd_data: dict):
        """Discretize the pressure gradient term of the Biot equation.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            sd_data: For entries, see above.

        Raises:
            NotImplementedError, the discretization should be performed using the
            discretize method of the Biot class.

        """
        raise NotImplementedError(
            """No discretize method implemented for the GradP class. See the Biot
            class."""
        )

    def assemble_matrix_rhs(
        self, sd: pp.Grid, sd_data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side for a discretization of the pressure
        gradient term of the Biot equation.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            sd_data: dictionary to store the data. For details on necessary keywords,
                see method discretize().

        Returns:
            matrix: sparse csr (sd.dim * g_num_cells, sd.dim * sd.num_cells).
                Discretization matrix.
            rhs: array (sd.dim * sd.num_cells) Right-hand side.

        """
        mat_dict = sd_data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary: dict = sd_data[pp.PARAMETERS][self.keyword]

        mat_key = self.grad_p_matrix_key
        if mat_key not in mat_dict.keys():
            raise ValueError(
                """GradP class requires a pre-computed discretization to be
                             stored in the matrix dictionary."""
            )
        div = pp.fvutils.vector_divergence(sd)

        # Assemble matrix.
        if mat_dict[mat_key].shape[0] != sd.dim * sd.num_faces:
            hf2f_nd = pp.fvutils.map_hf_2_f(sd=sd)
            grad_p = hf2f_nd * mat_dict[mat_key]
        else:
            grad_p = mat_dict[mat_key]
        matrix = div * grad_p

        # Assemble right-hand side. The rhs contribution is the lhs discretization
        # applied to the reference value.
        p_reference = parameter_dictionary["p_reference"]
        rhs = div * grad_p * p_reference

        return matrix, rhs


class DivU(Discretization):
    """Class for the displacement divergence term of the Biot equation."""

    def __init__(
        self,
        mechanics_keyword: str = "mechanics",
        flow_keyword: str = "flow",
        variable: str = "displacement",
        mortar_variable: str = "mortar_displacement",
    ) -> None:
        """Set the mechanics keyword and specify the variables.

        The keywords are used to access and store parameters and discretization
        matrices.
        The variable names are used to obtain the previous solution for the time
        discretization. Consequently, they are those of the unknowns contributing to
        the DivU term (displacements), not the scalar variable.
        """
        self.flow_keyword = flow_keyword
        self.mechanics_keyword = mechanics_keyword

        # Use the same key to access the discretization matrix as the Biot class.
        self.div_u_matrix_key = Biot().div_u_matrix_key
        self.bound_div_u_matrix_key = Biot().bound_div_u_matrix_key
        # We also need to specify the names of the displacement variables on the node
        # and adjacent edges.
        # Set variable name for the vector variable (displacement).
        self.variable = variable
        # The following is only used for mixed-dimensional problems.
        # Set the variable used for contact mechanics.
        self.mortar_variable = mortar_variable

    def ndof(self, sd: pp.Grid):
        """Return the number of degrees of freedom associated to the method.

        Parameters:
            sd: A grid.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_cells

    def discretize(self, sd: pp.Grid, sd_data: dict):
        """Discretize the displacement divergence term of the Biot equation.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            sd_data: For entries, see above.

        Raises:
            NotImplementedError, the discretization should be performed using the
            discretize method of the Biot class.
        """
        raise NotImplementedError(
            """No discretize method implemented for the GradP
                                  class. See the Biot class."""
        )

    def assemble_matrix_rhs(
        self, sd: pp.Grid, sd_data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side for a discretization of the
        displacement divergence term of the Biot equation.

        Parameters:
            sd : grid, or a subclass, with geometry fields computed.
            sd_data: dictionary to store the data. For details on necessary keywords,
                see method discretize()

        Returns:
            matrix: sparse csr (sd.dim * g_num_cells, sd.dim * g_num_cells)
                Discretization matrix.
            rhs: array (sd.dim * g_num_cells) Right-hand side.

        """
        matrix_dictionary = sd_data[pp.DISCRETIZATION_MATRICES][self.flow_keyword]
        parameter_dictionary_mech = sd_data[pp.PARAMETERS][self.mechanics_keyword]
        parameter_dictionary_flow = sd_data[pp.PARAMETERS][self.flow_keyword]

        mat_key = self.div_u_matrix_key
        if mat_key not in matrix_dictionary.keys():
            raise ValueError(
                """DivU class requires a pre-computed discretization to be
                             stored in the matrix dictionary."""
            )
        # Assemble matrix.
        biot_alpha = parameter_dictionary_flow["biot_alpha"]
        matrix = matrix_dictionary[mat_key] * biot_alpha

        # Assemble right-hand side.
        # For IE and constant BCs, the boundary part cancels, as the contribution from
        # successive time steps (n and n+1) appear on the rhs with opposite signs. For
        # transient BCs, use the below with the appropriate version of d_bound_i.
        # Get bc values from mechanics
        d_bound_1 = parameter_dictionary_mech["bc_values"]
        d_bound_0 = parameter_dictionary_mech["bc_values_previous_timestep"]
        # and coupling parameter from flow
        rhs_bound = (
            -matrix_dictionary[self.bound_div_u_matrix_key]
            * (d_bound_1 - d_bound_0)
            * biot_alpha
        )

        # Time part
        d_cell = pp.get_solution_values(
            name=self.variable, data=sd_data, time_step_index=0
        )
        div_u = matrix_dictionary[mat_key]
        rhs_time = np.squeeze(biot_alpha * div_u * d_cell)
        return matrix, rhs_bound + rhs_time


class BiotStabilization(Discretization):
    """Class for the stabilization term of the Biot equation."""

    def __init__(self, keyword="mechanics", variable="pressure"):
        """Set the two keywords.

        The keywords are used to access and store parameters and discretization
        matrices.
        """
        self.keyword = keyword
        # Set variable name for the scalar variable (pressure)
        self.variable = variable

        # Use same keyword as in Biot class
        self.stabilization_matrix_key = Biot().stabilization_matrix_key

    def ndof(self, sd: pp.Grid):
        """Return the number of degrees of freedom associated to the method.

        Parameters:
            sd: A grid.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_cells

    def discretize(self, sd: pp.Grid, sd_data: dict):
        """Discretize the stabilization term of the Biot equation.

        Parameters:
            g (pp.Grid): grid, or a subclass, with geometry fields computed.
            sd_data: Data dictionary.

        Raises:
            NotImplementedError, the discretization should be performed using the
            discretize method of the Biot class.

        """
        raise NotImplementedError(
            """No discretize method implemented for the BiotStabilization
                                  class. See the Biot class."""
        )

    def assemble_matrix_rhs(
        self, sd: pp.Grid, sd_data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side for a discretization of the
        stabilization term of the Biot equation.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            sd_data: dictionary to store the data. For details on necessary keywords,
                see method discretize().

        Returns:
            matrix: sparse csr (sd.dim * g.num_cells, sd.dim * sd.num_cells)
                Discretization matrix.
            rhs: array (sd.dim * sd.num_cells) Right-hand side.

        """
        mat_key = self.stabilization_matrix_key
        matrix_dictionary = sd_data[pp.DISCRETIZATION_MATRICES][self.keyword]

        if mat_key not in matrix_dictionary.keys():
            raise ValueError(
                """BiotStabilization class requires a pre-computed
                             discretization to be stored in the matrix dictionary."""
            )

        # The stabilization is the pressure contribution to the div u part of the
        # fluid mass conservation, thus needs a right-hand side in the implicit Euler
        # discretization.
        pressure_0 = pp.get_solution_values(
            name=self.variable, data=sd_data, time_step_index=0
        )
        matrix = matrix_dictionary[self.stabilization_matrix_key]
        rhs_time = matrix * pressure_0

        return matrix, rhs_time
