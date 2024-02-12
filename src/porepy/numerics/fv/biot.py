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

    ``data[pp.DISCRETIZATION_MATRICES][self.keyword]["grad_p"]``:
    Discretization of the term :math:`\\nabla p` in the mechanics part of the
    poromechanical system.

    ``data[pp.DISCRETIZATION_MATRICES][self.keyword]["bound_displacement_pressure"]``:
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
        keyword: str = "mechanics",
    ) -> None:
        """Set the two keywords.

        The keywords are used to access and store parameters and discretization
        matrices.
        """
        super().__init__("")

        self.keyword = keyword
        """Keyword used to identify the parameter dictionary associated with the
        mechanics subproblem."""

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
        # These are moved to the self.keyword matrix dictionary, and will then be
        # moved back again at the end of this function
        mech_in_flow = [
            self.div_u_matrix_key,
            self.bound_div_u_matrix_key,
            self.stabilization_matrix_key,
        ]
        for key in mech_in_flow:
            mat = sd_data[pp.DISCRETIZATION_MATRICES][self.flow_keyword].pop(key)
            sd_data[pp.DISCRETIZATION_MATRICES][self.keyword][key] = mat

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
            self.keyword,
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
            sd_data[pp.DISCRETIZATION_MATRICES][self.keyword].pop(key, None)

    def discretize(self, sd: pp.Grid, sd_data: dict) -> None:
        """Discretize the mechanics terms in a poromechanical system.

        NOTE: This function does *not* discretize purely flow-related terms (Darcy flow
        and compressibility).

        The parameters needed for the discretization are stored in the dictionary
        sd_data, which should contain the following mandatory keywords:

            Related to mechanics equation (in
            sd_data[pp.PARAMETERS][self.keyword]):
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
            self.keyword
        ]

        bound: pp.BoundaryConditionVectorial = parameter_dictionary["bc"]
        constit: pp.FourthOrderTensor = parameter_dictionary["fourth_order_tensor"]

        eta: float = parameter_dictionary.get("mpsa_eta", pp.fvutils.determine_eta(sd))
        inverter: Literal["python", "numba"] = parameter_dictionary.get(
            "inverter", "numba"
        )

        scalar_vector_mappings: dict = parameter_dictionary["scalar_vector_mappings"]
        coupling_keywords: list[str] = list(scalar_vector_mappings.keys())

        alpha: dict[str, pp.SecondOrderTensor] = {}

        for key, alpha_input in scalar_vector_mappings.items():
            # TODO: Revisit 'biot_coupling_coefficient
            if isinstance(alpha_input, (float, int)):
                alpha[key] = pp.SecondOrderTensor(alpha_input * np.ones(sd.num_cells))
            else:
                alpha[key] = alpha_input

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
        active_constit: pp.FourthOrderTensor = (
            pp.fvutils.restrict_fourth_order_tensor_to_subgrid(constit, active_cells)
        )
        active_alpha: dict[str, pp.SecondOrderTensor] = {}
        for key, val in alpha.items():
            active_alpha[key] = pp.fvutils.restrict_second_order_tensor_to_subgrid(
                alpha[key], active_cells
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

        active_bound_displacement_cell = sps.csr_matrix((nf * nd, nc * nd))
        active_bound_displacement_face = sps.csr_matrix((nf * nd, nf * nd))
        active_bound_displacement_pressure = sps.csr_matrix((nf * nd, nc))

        active_grad_p, active_div_u, active_bound_div_u, active_stabilization = (
            {},
            {},
            {},
            {},
        )
        for key in coupling_keywords:
            active_grad_p[key] = sps.csr_matrix((nf * nd, nc))
            active_div_u[key] = sps.csr_matrix((nc, nc * nd))
            active_bound_div_u[key] = sps.csr_matrix((nc, nf * nd))
            active_stabilization[key] = sps.csr_matrix((nc, nc))

        def matrices_from_dict(d: dict) -> list:
            return [mat for mat in d.values()]

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
            loc_c: pp.FourthOrderTensor = (
                pp.fvutils.restrict_fourth_order_tensor_to_subgrid(
                    active_constit, l2g_cells
                )
            )
            # Copy Biot coefficient, and restrict to local cells
            loc_alpha = {}
            for key in active_alpha:
                loc_alpha[key] = pp.fvutils.restrict_second_order_tensor_to_subgrid(
                    active_alpha[key], l2g_cells
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
                *matrices_from_dict(loc_grad_p),
                loc_bound_displacement_pressure,
            )

            eliminate_cell = np.where(
                np.logical_not(np.in1d(l2g_cells, cells_in_subgrid))
            )[0]
            pp.fvutils.remove_nonlocal_contribution(
                eliminate_cell,
                1,
                *matrices_from_dict(loc_div_u),
                *matrices_from_dict(loc_bound_div_u),
                *matrices_from_dict(loc_biot_stab),
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

            for key in coupling_keywords:
                active_grad_p[key] += face_map_vec * loc_grad_p[key] * cell_map_scalar
                active_div_u[key] += (
                    cell_map_scalar.transpose() * loc_div_u[key] * cell_map_vec
                )
                active_bound_div_u[key] += (
                    cell_map_scalar.transpose()
                    * loc_bound_div_u[key]
                    * face_map_vec.transpose()
                )
                active_stabilization[key] += (
                    cell_map_scalar.transpose() * loc_biot_stab[key] * cell_map_scalar
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
        # IMPLEMENTATION NOTE: This scaling is only needed for the vector quantities,
        # that is, not for the stabilization and div_u terms. The latter are computed
        # cell-wise rather than face-wise, and will only discretized once, even for
        # cells next to subdomain boundaries.
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
        active_bound_displacement_pressure = (
            scaling_vector @ active_bound_displacement_pressure
        )
        for key in coupling_keywords:
            active_grad_p[key] = scaling_vector @ active_grad_p[key]

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

        # Update coupling terms, stored as dictionaries
        grad_p, div_u, bound_div_u, stabilization = {}, {}, {}, {}
        for key in coupling_keywords:
            grad_p[key] = face_map_vec * active_grad_p[key] * cell_map_scalar
            div_u[key] = (
                cell_map_scalar.transpose() * active_div_u[key] * cell_map_vec
            ).tocsr()
            bound_div_u[key] = (
                cell_map_scalar.transpose()
                * active_bound_div_u[key]
                * face_map_vec.transpose()
            ).tocsr()
            stabilization[key] = (
                cell_map_scalar.transpose()
                * active_stabilization[key]
                * cell_map_scalar
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
            *matrices_from_dict(grad_p),
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
            eliminate_cells,
            1,
            *matrices_from_dict(div_u),
            *matrices_from_dict(bound_div_u),
            *matrices_from_dict(stabilization),
        )

        # Either update the discretization scheme, or store the full one
        matrices: dict[str, dict] = sd_data[pp.DISCRETIZATION_MATRICES]
        matrices_m: dict[str, sps.spmatrix] = matrices[self.keyword]

        if update:
            # The faces to be updated are given by active_faces
            update_face_ind = pp.fvutils.expand_indices_nd(active_faces, sd.dim)

            matrices_m[self.stress_matrix_key][update_face_ind] = stress[
                update_face_ind
            ]
            matrices_m[self.bound_stress_matrix_key][update_face_ind] = bound_stress[
                update_face_ind
            ]
            
            matrices_m[self.div_u_matrix_key][update_cell_ind] = div_u[update_cell_ind]
            matrices_m[self.bound_div_u_matrix_key][
                update_cell_ind
            ] = bound_div_u[update_cell_ind]
            matrices_m[self.grad_p_matrix_key][update_face_ind] = grad_p[update_face_ind]
            matrices_m[self.stabilization_matrix_key][
                update_cell_ind
            ] = stabilization[update_cell_ind]

            matrices_m[self.bound_displacement_cell_matrix_key][
                update_face_ind
            ] = bound_displacement_cell[update_face_ind]
            matrices_m[self.bound_displacement_face_matrix_key][
                update_face_ind
            ] = bound_displacement_face[update_face_ind]
            matrices_m[self.bound_pressure_matrix_key][
                update_face_ind
            ] = bound_displacement_pressure[update_face_ind]
        else:
            matrices_m[self.stress_matrix_key] = stress
            matrices_m[self.bound_stress_matrix_key] = bound_stress
            matrices_m[
                self.bound_displacement_cell_matrix_key
            ] = bound_displacement_cell
            matrices_m[
                self.bound_displacement_face_matrix_key
            ] = bound_displacement_face
            matrices_m[self.bound_pressure_matrix_key] = bound_displacement_pressure
            matrices_m[self.grad_p_matrix_key] = grad_p
            matrices_m[self.div_u_matrix_key] = div_u
            matrices_m[self.bound_div_u_matrix_key] = bound_div_u
            matrices_m[self.stabilization_matrix_key] = stabilization

    def _local_discretization(
        self,
        sd: pp.Grid,
        constit: pp.FourthOrderTensor,
        bound_mech: pp.BoundaryConditionVectorial,
        alpha: list[pp.SecondOrderTensor],
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

        # The boundary discretization of the div_u term is represented directly
        # on the cells, instead of going via the faces.
        grad_p, bound_div_u, div_u, stabilization = {}, {}, {}, {}
        for key in alpha:
            # trace of strain matrix
            div = self._subcell_gradient_to_cell_scalar(
                sd, cell_node_blocks, alpha[key], igrad
            )
            div_u[key] = div * igrad * rhs_cells
            bound_div_u[key] = div * igrad * rhs_bound

            # Call discretization of grad_p-term
            rhs_jumps, grad_p_face = self._create_rhs_grad_p(
                sd, subcell_topology, alpha[key], bound_exclusion_mech
            )

            if hf_output:
                # If boundary conditions are given on subfaces we keep the subface
                # discretization
                raise NotImplementedError("This should be deprecated")
            else:
                # otherwise, we map it to faces
                grad_p[key] = hf2f * (hook * igrad * rhs_jumps + grad_p_face)

            # consistency term for the flow equation
            stabilization[key] = div * igrad * rhs_jumps

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

    def _create_rhs_grad_p(
        self,
        sd: pp.Grid,
        subcell_topology: pp.fvutils.SubcellTopology,
        alpha: pp.SecondOrderTensor,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
    ) -> tuple[sps.spmatrix, sps.spmatrix]:
        """Consistent discretization of grad_p-term in MPSA-W method.

        Parameters:
            sd: grid to be discretized.
            subcell_topology: Wrapper class for numbering of subcell faces, cells
                etc.
            alpha: Biot's coupling coefficient, given as a tensor.
            bound_exclusion: Object that can eliminate faces related to boundary
                conditions.

        Returns:
            scipy.sparse.csr_matrix ``shape: (num_subcells * dim, num_cells)``:
                discretization of the jumps in [n alpha p] term, ready to be multiplied
                with inverse gradient
            scipy.sparse.csr_matrix, ``shape (num_subfaces * dim, num_cells)``:
                discretization of the force on the face due to cell-centre
                pressure from a unique side.

        """
        # Background/motivation: In the extension of MPSA from pure elasticity to
        # poroelasticity (or thermoelasticity etc.), the consistent discretization of
        # the pressure force, represented as ``alpha p`` where ``alpha`` is a 2-tensor
        # consists of two terms (to see this, see sections 3 and 4 in Nordbotten 2016 -
        # but be aware a careful read will be needed): First, the pressure force is
        # computed on a face as ``n\dot (alpha p)`` where n is the normal vector; this
        # is the variable grad_p_face below. Second, imbalances in the pressure force
        # between neighboring cells give rise to local mechanical deformation, which
        # again induce stresses; this is variable rhs_jumps below.  The discretization
        # of the first of these terms is computed in this helper method, while for the
        # second, we construct a rhs-matrix (in the parlance of the Mpsa implementation)
        # that will be multiplied with the inverse deformation gradient elsewhere.
        #
        # The implementation is divided into three main steps:
        # 1. compute product normal_vector * alpha and get a map for vector problems
        # 2. assemble r.h.s. for the new linear system, needed for the term
        #    'rhs_jumps'
        # 3. compute term 'grad_p_face'
        
        # Step 1: Compute normal vectors * alpha
        nd = sd.dim
        num_subhfno = subcell_topology.subhfno.size
        num_subfno_unique = subcell_topology.num_subfno_unique
        num_subfno = subcell_topology.num_subfno

        if nd == 2:
            # For 2d, we need to remove the z-component of alpha to ensure that the
            # tensor has as many components as there are spatial dimensions. Do this on
            # a copy to avoid changing the input.
            alpha = alpha.copy()
            alpha.values = np.delete(alpha.values, (2), axis=0)
            alpha.values = np.delete(alpha.values, (2), axis=1)

        # Obtain normal_vector * alpha. nAlpha_grad will have separate rows for each
        # sub-halfface, that is, the right and left side of an internal face will be
        # assigned separate rows.
        (
            nAlpha_grad,
            cell_node_blocks,
            sub_cell_index,
        ) = pp.fvutils.scalar_tensor_vector_prod(sd, alpha, subcell_topology)
        # Transfer nAlpha to a subface-based quantity by pairing expressions on the two
        # sides of the subface. That is, for internal faces, the elements corresponding
        # to the left and right side of the face will be put on the same row.
        unique_nAlpha_grad = subcell_topology.pair_over_subfaces(nAlpha_grad)

        def component_wise_ordering(mat: sps.spmatrix, nd: int, ind: np.ndarray) -> sps.spmatrix:
            # Convenience method for reshaping nAlpha from face-based to
            # component-based. This is to build a block diagonal sparse matrix
            # compatible with igrad * rhs_units, that is first all x-component, then y,
            # and z
            return sps.block_diag([mat[:, ind[i]] for i in range(nd)], format='csr')

        # Reshape nAlpha component-wise
        nAlpha_grad = component_wise_ordering(nAlpha_grad, nd, sub_cell_index)
        unique_nAlpha_grad = component_wise_ordering(unique_nAlpha_grad, nd, sub_cell_index)

        # Step 2: Compute rhs_jumps. This term represents the force imbalance caused by
        # the cell center pressures in the adjacent cells. The goal of the next lines is
        # to construct a right-hand side term that can be multiplied with the inverse
        # gradient (elsewhere) to obtain the mechanical deformation. The imbalance is
        # calculated from the terms unique_nAlpha_grad, which gives a pressure force on
        # the subfaces as a function of cell center pressures. To be compatible with the
        # wider discretization approach, unique_nAlpha_grad must 1) be reordered to
        # match the ordering of the local equations, and 2) be multiplied with a
        # mapping from cell center pressures to subface forces.

        # Construct a mapping from the ordering of unique_nAlpha_grad to the ordering of
        # the local equations. To that end, recall the ordering of the local equations
        # (see self._local_discretization()):
        # i) stress equilibrium for the internal subfaces.
        # ii) the stress equilibrium for the Neumann subfaces.
        # iii) the Robin subfaces.
        # iv) the displacement continuity on both internal and external subfaces.

        # Construct a diagonal matrix with the same number of rows as
        # unique_nAlpha_grad. We will pick out rows corresponding to faces in the
        # interior, and various types of boundary conditions, and construct the
        # requested mapping.
        sz = nd * num_subfno_unique
        rhs_units = sps.dia_matrix((np.ones(sz), 0), shape=(sz,sz))

        # Divide into internal and external faces. On internal faces there is no need to
        # account for the sign of the normal vector - this is encoded in nAlpha_grad.
        rhs_int = bound_exclusion.exclude_boundary(rhs_units)
        rhs_neu = bound_exclusion.keep_neumann(rhs_units)
        rhs_rob = bound_exclusion.keep_robin(rhs_units)

        num_dir_subface = (
            bound_exclusion.exclude_neu_rob.shape[1]
            - bound_exclusion.exclude_neu_rob.shape[0]
        )
        # No right-hand side for cell displacement equations.
        rhs_units_displ_var = sps.csr_matrix(
            (nd * num_subfno - num_dir_subface, num_subfno_unique * nd)
        )

        # The row mapping
        row_mapping = sps.vstack([rhs_int, rhs_neu, rhs_rob, rhs_units_displ_var])

        # Mapping from scalar cell values (e.g., the pressure as a potential) to subcell
        # vector quantities (pressure as a force on subcells).
        sc2c = pp.fvutils.cell_scalar_to_subcell_vector(
            sd.dim, sub_cell_index, cell_node_blocks[0]
        )
        # The representation of unique_nAlpha_grad, ready to be multiplied with the
        # inverse gradient.
        rhs_jumps = row_mapping * unique_nAlpha_grad * sc2c

        # Step 3: Compute grad_p_face. This term represents the force on the face due to
        # cell-centre pressure from a unique side. To that end, construct a mapping that
        # keeps all boundary faces, but only one side of the internal faces. Note that
        # this mapping acts on nAlpha_grad, not unique_nAlpha_grad, that is a row
        # contains only the force on one side of the face.
        vals = np.ones(num_subfno_unique * nd)
        rows = pp.fvutils.expand_indices_nd(subcell_topology.subfno_unique, nd)
        cols = pp.fvutils.expand_indices_incr(
            subcell_topology.unique_subfno, nd, num_subhfno
        )
        map_unique_subfno = sps.coo_matrix(
            (vals, (rows, cols)), shape=(num_subfno_unique * nd, num_subhfno * nd)
        ).tocsr()

        # Prepare for computation of -grad_p_face term
        grad_p_face = - map_unique_subfno * nAlpha_grad * sc2c

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
        self,
        sd: pp.Grid,
        cell_node_blocks: np.ndarray,
        alpha: pp.SecondOrderTensor,
        igrad,
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

        inds = np.arange(nd**2)

        # Sub-cell wise trace of strain tensor: One row per sub-cell

        row, col = np.meshgrid(np.arange(cell_node_blocks.shape[1]), inds)
        # Adjust the columns to hit each sub-cell
        incr = np.cumsum(nd**2 * np.ones(cell_node_blocks.shape[1])) - nd**2
        col += incr.astype("int32")

        # Distribute the values of the alpha tensor to the subcells by using
        # cell_node_blocks. For 2d grids, also drop the z-components of the alpha tensor.
        # NOTE: This assumes that for a 2d grid, the grid is located in the xy-plane.
        # This is a tacit assumption throughout the Biot and Mpsa implementations (but
        # not mpfa, see the use of a rotation matrix in that module).
        subcell_alpha_values = alpha.values[:nd, :nd, cell_node_blocks[0]]

        # Reorder the elements of the subcell_alpha_values (which is a 3-tensor), so
        # that the elements of the alpha tensor are ordered as (in 2d, the 3d extension
        # is the natural one; also subscripts give x and y index, superscript refers to
        # the ordering of cells in subcell_alpha_values)
        #
        #   [alpha_11^0, alpha_12^0, alpha_21^0, alpha_22^0, alpha_11^1, ...]
        #
        # To understand (or perhaps better, to accept) the below code, it is useful to
        # consider the following example:
        #
        # >> a = np.arange(24).reshape((2, 3, 4))
        # >> a[:, :, 0]
        #    array([[ 0,  4,  8],
        #           [12, 16, 20]])
        # >> a[:, :, 1]
        #    array([[ 1,  5,  9],
        #           [13, 17, 21]])
        #
        # The goal is to run through the elements of a[:, :, 0], then a[:, :, 1] etc.
        # Some trial and error showed that the following works:
        #
        # >> a.swapaxes(2, 1).swapaxes(1, 0).ravel()
        #    array([ 0,  4,  8, 12, 16, 20,  1,  5,  9, 13, 17, 21,  2,  6, 10, 14, 18,
        #           22,  3,  7, 11, 15, 19, 23])
        #
        # Fingers crossed we never need to revisit this part.
        subcell_alpha_reordered = (
            subcell_alpha_values.swapaxes(2, 1).swapaxes(1, 0).ravel()
        )

        # Integrate the trace over the sub-cell, that is, distribute the cell
        # volumes equally over the sub-cells
        num_cell_nodes = sd.num_cell_nodes()
        cell_vol = sd.cell_volumes / num_cell_nodes
        factor = np.repeat(cell_vol[cell_node_blocks[0]], nd**2)
        val = factor * subcell_alpha_reordered
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
        matrix_dictionary = sd_data[pp.DISCRETIZATION_MATRICES][self.keyword]
        stress_discr = matrix_dictionary["stress"]
        bound_stress = matrix_dictionary["bound_stress"]
        bound_val = sd_data[pp.PARAMETERS][self.keyword]["bc_values"]
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
        self.keyword = mechanics_keyword

        # Use the same key to access the discretization matrix as the Biot class.
        biot_discr = Biot(mechanics_keyword=mechanics_keyword)
        self.div_u_matrix_key = biot_discr.div_u_matrix_key
        self.bound_div_u_matrix_key = biot_discr.bound_div_u_matrix_key
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
        parameter_dictionary_mech = sd_data[pp.PARAMETERS][self.keyword]
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
        self.stabilization_matrix_key = Biot([]).stabilization_matrix_key

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
