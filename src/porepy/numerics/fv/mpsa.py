"""
Implementation of the multi-point stress appoximation method.

The implementation is based on the weakly symmetric version of MPSA, described in

    Keilegavlen, Nordbotten: Finite volume methods for elasticity with weak symmetry,
        IJNME, 2017.

"""
from __future__ import annotations

import logging
from time import time
from typing import Any, Literal, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization

# Module-wide logger
logger = logging.getLogger(__name__)


class Mpsa(Discretization):
    """Implementation of the Multi-point stress approximation.

    Attributes:
        keyword (str): Keyword used to identify the parameter dictionary.
            Defaults to "mechanics".
        stress_matrix_key (str): Keyword used to identify the discretization matrix for
            the stress. Defaults to "stress".
        bound_stress_matrix_key (str): Keyword used to identify the discretization
             matrix for the boundary conditions for stress. Defaults to "bound_stress".
        bound_displacement_cell_matrix_key (str): Keyword used to identify the
            discretization matrix for the cell center displacement contribution to
            boundary displacement reconstrution. Defaults to "bound_displacement_cell".
        bound_displacement_face_matrix_key (str): Keyword used to identify the
            discretization matrix for the boundary conditions' contribution to
            boundary displacement reconstrution. Defaults to "bound_displacement_face".

    """

    def __init__(self, keyword: str) -> None:
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

        self.stress_matrix_key = "stress"
        self.bound_stress_matrix_key = "bound_stress"
        self.bound_displacement_cell_matrix_key = "bound_displacement_cell"
        self.bound_displacement_face_matrix_key = "bound_displacement_face"

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, sd: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells times dimension (stress dof).

        Parameter
        ---------
        sd: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return sd.dim * sd.num_cells

    def extract_displacement(
        self, sd: pp.Grid, solution_array: np.ndarray, d: dict
    ) -> np.ndarray:
        """Extract the displacement part of a solution.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid.
            d (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.

        Returns:
            np.array (sd.num_cells): Displacement solution vector. Will be identical
                to solution_array.

        """
        return solution_array

    def extract_stress(
        self, sd: pp.Grid, solution_array: np.ndarray, d: dict
    ) -> np.ndarray:
        """Extract the stress corresponding to a solution

        The stress is composed of contributions from the solution variable and the
        boundary conditions.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid.
            d (dictionary): Data dictionary associated with the grid.

        Returns:
            np.array (sd.num_cells): Vector of stresses on the grid faces.

        """
        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = d[pp.PARAMETERS][self.keyword]

        stress = matrix_dictionary[self.stress_matrix_key].tocsr()
        bound_stress = matrix_dictionary[self.bound_stress_matrix_key].tocsr()

        bc_val = parameter_dictionary["bc_values"]

        return stress * solution_array + bound_stress * bc_val

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """
        Discretize the second order vector elliptic equation using multi-point stress
        approximation.

        The method computes traction over faces in terms of cell center displacements.

        It is possible to do a partial discretization via parameters specified_cells,
        _nodes and _faces. This is considered an advanced, and only partly tested
        option.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in ``data[pp.PARAMETERS][self.keyword]``.
            matrix_dictionary, for storage of discretization matrices.
                Stored in ``data[pp.DISCRETIZATION_MATRICES][self.keyword]``

        parameter_dictionary contains the entries:
            - fourth_order_tensor: ``class:~porepy.params.tensor.FourthOrderTensor``
                Stiffness tensor defined cell-wise.
            - bc: ``class:~porepy.params.bc.BoundaryConditionVectorial``
                Boundary conditions
            - mpsa_eta: ``float`` Optional. Range [0, 1). Location of displacement
                continuity point: eta. ``eta = 0`` gives cont. pt. at face midpoint,
                ``eta = 1`` at the vertex. If not given, PorePy tries to set an optimal
                value. This value is set to all subfaces, except the boundary (where,
                0 is used).
            - inverter (``str``): Optional. Inverter to apply for local problems.
                Can take values 'numba' (default), or 'python'.

        matrix_dictionary will be updated with the following entries:
            - ``stress: sps.csc_matrix (sd.dim * sd.num_faces, sd.dim * sd.num_cells)``
                stress discretization, cell center contribution
            - ``bound_flux: sps.csc_matrix (sd.dim * sd.num_faces, sd.dim *
                sd.num_faces)`` stress discretization, face contribution
            - ``bound_displacement_cell: sps.csc_matrix (sd.dim * sd.num_faces,
                                                     sd.dim * sd.num_cells)``
                Operator for reconstructing the displacement trace. Cell center
                contribution.
            - ``bound_displacement_face: sps.csc_matrix (sd.dim * sd.num_faces,
                                                     sd.dim * sd.num_faces)``
                Operator for reconstructing the displacement trace. Face contribution.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            data: For entries, see above.

        """
        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]
        constit: pp.FourthOrderTensor = parameter_dictionary["fourth_order_tensor"]
        bound: pp.BoundaryConditionVectorial = parameter_dictionary["bc"]

        eta: Optional[float] = parameter_dictionary.get("mpsa_eta", None)
        hf_eta: Optional[float] = parameter_dictionary.get("reconstruction_eta", None)

        inverter: Literal["python", "numba"] = parameter_dictionary.get(
            "inverter", "numba"
        )
        max_memory: int = parameter_dictionary.get("max_memory", 1e9)

        # Whether to update an existing discretization, or construct a new one. If True,
        # either specified_cells, _faces or _nodes should also be given, or else a full
        # new discretization will be computed
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

        # Bookkeeping.
        nd = active_grid.dim
        nf = active_grid.num_faces
        nc = active_grid.num_cells

        # Empty matrices for stress, bound_stress and boundary displacement
        # reconstruction. Will be expanded as we go.
        # Implementation note: It should be relatively straightforward to estimate the
        # memory need of stress (face_nodes -> node_cells -> unique).
        active_stress = sps.csr_matrix((nf * nd, nc * nd))
        active_bound_stress = sps.csr_matrix((nf * nd, nf * nd))
        active_bound_displacement_cell = sps.csr_matrix((nf * nd, nc * nd))
        active_bound_displacement_face = sps.csr_matrix((nf * nd, nf * nd))

        # Find an estimate of the peak memory need
        peak_memory_estimate = self._estimate_peak_memory_mpsa(active_grid)

        # Loop over all partition regions, construct local problems, and transfer
        # discretization to the entire active grid.
        for reg_i, (sub_g, faces_in_subgrid, _, l2g_cells, l2g_faces) in enumerate(
            pp.fvutils.subproblems(active_grid, max_memory, peak_memory_estimate)
        ):
            tic = time()

            # Copy stiffness tensor, and restrict to local cells.
            loc_c: pp.FourthOrderTensor = self._constit_for_subgrid(
                active_constit, l2g_cells
            )

            # Boundary conditions are slightly more complex. Find local faces that are
            # on the global boundary. Then transfer boundary condition on those faces.
            loc_bnd: pp.BoundaryConditionVectorial = self._bc_for_subgrid(
                active_bound, sub_g, l2g_faces
            )

            # Discretization of sub-problem
            (
                loc_stress,
                loc_bound_stress,
                loc_bound_displacement_cell,
                loc_bound_displacement_face,
            ) = self._stress_disrcetization(
                sub_g, loc_c, loc_bnd, eta=eta, inverter=inverter, hf_eta=hf_eta
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
            )

            # Next, transfer discretization matrices from the local to the active grid
            # Get a mapping from the local to the active grid
            face_map, cell_map = pp.fvutils.map_subgrid_to_grid(
                active_grid, l2g_faces, l2g_cells, is_vector=True
            )

            # Update discretization on the active grid.
            active_stress += face_map * loc_stress * cell_map
            active_bound_stress += face_map * loc_bound_stress * face_map.transpose()

            # Update global face fields.
            active_bound_displacement_cell += (
                face_map * loc_bound_displacement_cell * cell_map
            )
            active_bound_displacement_face += (
                face_map * loc_bound_displacement_face * face_map.transpose()
            )
            logger.info(f"Done with subproblem {reg_i}. Elapsed time {time() - tic}")

        # We have reached the end of the discretization, what remains is to map the
        # discretization back from the active grid to the entire grid
        face_map, cell_map = pp.fvutils.map_subgrid_to_grid(
            sd, extracted_faces, active_cells, is_vector=True
        )

        # Update global face fields.
        stress_glob = face_map * active_stress * cell_map
        bound_stress_glob = face_map * active_bound_stress * face_map.transpose()
        bound_displacement_cell_glob = (
            face_map * active_bound_displacement_cell * cell_map
        )
        bound_displacement_face_glob = (
            face_map * active_bound_displacement_face * face_map.transpose()
        )

        eliminate_faces = np.setdiff1d(np.arange(sd.num_faces), active_faces)
        pp.fvutils.remove_nonlocal_contribution(
            eliminate_faces,
            sd.dim,
            stress_glob,
            bound_stress_glob,
            bound_displacement_cell_glob,
            bound_displacement_face_glob,
        )

        if update:
            update_ind = pp.fvutils.expand_indices_nd(active_faces, sd.dim)
            matrix_dictionary[self.stress_matrix_key][update_ind] = stress_glob[
                update_ind
            ]
            matrix_dictionary[self.bound_stress_matrix_key][
                update_ind
            ] = bound_stress_glob[update_ind]
            matrix_dictionary[self.bound_displacement_cell_matrix_key][
                update_ind
            ] = bound_displacement_cell_glob[update_ind]
            matrix_dictionary[self.bound_displacement_face_matrix_key][
                update_ind
            ] = bound_displacement_face_glob[update_ind]
        else:
            matrix_dictionary[self.stress_matrix_key] = stress_glob
            matrix_dictionary[self.bound_stress_matrix_key] = bound_stress_glob
            matrix_dictionary[
                self.bound_displacement_cell_matrix_key
            ] = bound_displacement_cell_glob
            matrix_dictionary[
                self.bound_displacement_face_matrix_key
            ] = bound_displacement_face_glob

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


        define cells, faces and nodes that have been modified (either parameters,
        geometry or topology), and should be rediscretized. It is up to the
        discretization method to implement the change necessary by this modification.
        Note that depending on the computational stencil of the discretization method,
        a grid quantity may be rediscretized even if it is not marked as modified.

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
            data: With discretization parameters.

        """
        # Keywords that should be interpreted as vector cell quantities
        vector_cell_right = [
            self.stress_matrix_key,
            self.bound_displacement_cell_matrix_key,
        ]
        vector_face_right = [
            self.bound_stress_matrix_key,
            self.bound_displacement_face_matrix_key,
        ]

        vector_face_left = [
            self.stress_matrix_key,
            self.bound_displacement_cell_matrix_key,
            self.bound_displacement_face_matrix_key,
            self.bound_stress_matrix_key,
        ]

        pp.fvutils.partial_update_discretization(
            sd,
            data,
            self.keyword,
            self.discretize,
            dim=sd.dim,
            vector_cell_right=vector_cell_right,
            vector_face_right=vector_face_right,
            vector_face_left=vector_face_left,
        )

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point stress
        approximation.

        Parameters:
            g: Grid to be discretized.
            data: dictionary to store the data. For details on necessary keywords,
                see ``:meth:discretize``.

        Returns:
            :obj:`~scipy.sparse.spmatrix`: ``(sd.dim * g_num_cells,
                                              sd.dim * g_num_cells)``

                    Discretization matrix.
            :obj:`~np.ndarray`: ``(sd.dim * g_num_cells)``

                Right-hand side which contains the boundary conditions and the vector
                source term.

        """
        return self.assemble_matrix(sd, data), self.assemble_rhs(sd, data)

    def assemble_matrix(self, sd: pp.Grid, data: dict) -> sps.spmatrix:
        """Return the matrix for a discretization of a second order elliptic vector
        equation using a FV method.

        Parameters:
            g: Grid to be discretized.
            data: dictionary to store the data. For details on necessary keywords,
                see ``:meth:discretize``.

        Returns
            :obj:`~scipy.sparse.spmatrix`: ``(sd.dim * g_num_cells, sd.dim * g_num_cells)``

                Discretization matrix.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        div = pp.fvutils.vector_divergence(sd)
        stress = matrix_dictionary["stress"]
        if stress.shape[0] != sd.dim * sd.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(sd=sd)
            stress = hf2f * stress
        M = div * stress

        return M

    def assemble_rhs(self, sd: pp.Grid, data: dict) -> np.ndarray:
        """Return the right-hand side for a discretization of a second order elliptic
        equation using a finite volume method.

        Parameters:
            g: Grid to be discretized.
            data: dictionary to store the data. For details on necessary keywords,
                see ``:meth:discretize``.

        Returns
            :obj:`~np.ndarray`: ``(sd.dim * g_num_cells)``

            Right-hand side which contains the boundary conditions and the vector
            source term.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        bound_stress = matrix_dictionary["bound_stress"]
        if bound_stress.shape[0] != sd.dim * sd.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(sd=sd)
            bound_stress = hf2f * bound_stress

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        bc_val = parameter_dictionary["bc_values"]

        div = pp.fvutils.vector_divergence(sd)

        return -div * bound_stress * bc_val + parameter_dictionary["source"]

    def _stress_disrcetization(
        self,
        sd: pp.Grid,
        constit: pp.FourthOrderTensor,
        bound: pp.BoundaryConditionVectorial,
        eta: Optional[float] = None,
        inverter: Literal["numba", "python"] = "numba",
        hf_disp: bool = False,
        hf_eta: Optional[float] = None,
    ) -> tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix, sps.spmatrix]:
        """
        Actual implementation of the MPSA W-method. To calculate the MPSA
        discretization on a grid, either call this method, or, to respect the
        privacy of this method, call the main mpsa method with no memory
        constraints.

        Implementation details:

        The displacement is discretized as a linear function on sub-cells (see reference
        paper). In this implementation, the displacement is represented by its cell
        center value and the sub-cell gradients.

        The method will give continuous stresses over the faces, and displacement
        continuity for certain points (controlled by the parameter eta). This can be
        expressed as a linear system on the form

            (i)   A * grad_u            = 0
            (ii)  B * grad_u + C * u_cc = 0
            (iii) 0            D * u_cc = I

        Here, the first equation represents stress continuity, and involves only the
        displacement gradients (grad_u). The second equation gives displacement
        continuity over cell faces, thus B will contain distances between cell centers
        and the face continuity points, while C consists of +- 1 (depending on which
        side the cell is relative to the face normal vector). The third equation
        enforces the displacement to be unity in one cell at a time. Thus (i)-(iii) can
        be inverted to express the displacement gradients as in terms of the cell center
        variables, that is, we can compute the basis functions on the sub-cells. Because
        of the method construction (again see reference paper), the basis function of a
        cell c will be non-zero on all sub-cells sharing a vertex with c. Finally, the
        fluxes as functions of cell center values are computed by insertion into Hook's
        law (which is essentially half of A from (i), that is, only consider
        contribution from one side of the face.

        Boundary values can be incorporated with appropriate modifications - Neumann
        conditions will have a non-zero right hand side for (i), Robin conditions will
        be on the form E * grad_u + F * u_cc = R, while Dirichlet gives a right hand
        side for (ii).

        In the implementation we will order the rows of the local system as follows;
        first enforce the force balance over the internal faces;
            T^L + T^R = 0.
        Then we will enforce the Neumann conditions
            T = T_NEUMAN,
        and the robin conditions
            T + robin_weight * U = R.
        The displacement continuity and Dirichlet conditions are comming last
            U^+ - U^- = 0.
        Note that for the Dirichlet conditions are not pulled out seperatly as the
        Neumann condition, mainly for legacy reasons. This meens that Dirichlet faces
        and internal faces are mixed together, decided by their face ordering.

        Parameters:
            g: Grid to be discretized.
            constit: Constitutive law for the rock.
            bound: Boundary conditions for the displacement.
            eta: Parameter controlling continuity of displacement. If None, a default
                value will be used, adapted to the grid type.
            inverter: Method to be used for inversion of local systems. Options are
                'numba' (default) and 'python'.
            hf_disp: If True, the displacement will be represented by subface values
                instead of cell values. This is not recommended, but kept for the time
                being for legacy reasons.
            hf_eta: If hf_disp is True, this parameter will be used instead of eta to
                control the continuity of the displacement.

        Returns:
            tuple of 4 sps.spmatrix:

            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces * sd.dim,
                                                     sd.num_cells * sd.dim))``

                Stree discretization matrix.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces * sd.dim,
                                                     sd.num_faces * sd.dim))``

                Boundary condition discretization matrix.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces * sd.dim,
                                                     sd.num_cells * sd.dim))``

                Matrix for cell-center displacement contribution to face displacement
                reconstruction.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.num_faces * sd.dim,
                                                     sd.num_faces * sd.dim))``

                Matrix for boundary condition contribution to face displacement
                reconstruction.

        """
        # Implementational note on boundary conditions: In Porepy we have defined nodes,
        # faces and cells in our grid. The mpsa/mpfa methods, however, needs access to
        # the local subgrids around each node (see any mpfa paper) to formulate the
        # local problems. In Porepy we obtain these subgrids through the the class
        # SubcellTopology (in fvutils). Since the user usually don't have access to/care
        # about the subcells, she will define the boundary conditions on the faces of
        # the grid. But as you can see in the method documentation above, the boundary
        # conditions should in fact be applied to the subfaces. Porepy handles this by
        # distribute the given face-boundary condition to the sub-faces automatically.
        # In some cases, users that are experts in mpsa/mpsa might want to specify the
        # boundary conditions on the subfaces (e.g. to specify different values on each
        # subface or even different conditions). So far this is handled in the
        # implementation by checking the length of the boundary conditions to see if
        # they are the length of faces or subfaces. If the user gives a subface boundary
        # conditions, mpsa/mpfa will also return the discretization on a subface level
        # (instead of the usuall face-wise level). The intrigued user might find the
        # function fvutils.boundary_to_sub_boundary(..) helpfull. As a final note, the
        # continuity points defaults to the face centers (eta=0) on the boundary. This
        # will happen as long as a scalar eta is given. eta defaults to the face centers
        # on the boundary since most user will specify the boundary conditions on the
        # faces (thus, the face-centers). The expert user that specify subface boundary
        # conditions might also want eta!=0 also on the boundary. This is facilitated by
        # supplying the method with eta beeing a ndArray equal the number of subfaces
        # (SubcellTopology.num_subfno_unique) giving the eta value for each subface.

        if eta is None:
            eta = pp.fvutils.determine_eta(sd)

        if bound.bc_type != "vectorial":
            raise AttributeError("MPSA must be given a vectorial boundary condition")

        if hasattr(sd, "periodic_face_map"):
            raise NotImplementedError(
                "Periodic boundary conditions are not implemented for Mpsa"
            )

        if sd.dim == 1:
            tpfa_key = "tpfa_elasticity"
            discr = pp.Tpfa(tpfa_key)
            params = pp.Parameters(sd)

            # Implicitly set Neumann boundary conditions on the whole domain. More
            # general values should be permissible, but it will require handling of
            # rotated boundary conditions.
            if np.any(bound.is_dir):
                # T
                raise ValueError("have not considered Dirichlet boundary values here")

            bnd = pp.BoundaryCondition(sd)
            params["bc"] = bnd

            # The elasticity tensor here is set to 2*mu + lmbda, that is, the standard
            # diagonal term in the stiffness matrix
            k = pp.SecondOrderTensor(2 * constit.mu + constit.lmbda)
            params["second_order_tensor"] = k

            d: dict = {
                pp.PARAMETERS: {tpfa_key: params},
                pp.DISCRETIZATION_MATRICES: {tpfa_key: {}},
            }
            discr.discretize(sd, d)
            matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][tpfa_key]
            return (
                matrix_dictionary["flux"],
                matrix_dictionary["bound_flux"],
                matrix_dictionary["bound_pressure_cell"],
                matrix_dictionary["bound_pressure_face"],
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
            sd, constit = self._reduce_grid_constit_2d(sd, constit)

        nd = sd.dim

        # Define subcell topology
        subcell_topology = pp.fvutils.SubcellTopology(sd)
        # If g is not already a sub-grid we create one
        if bound.num_faces == subcell_topology.num_subfno_unique:
            subface_rhs = True
        else:
            # And we expand the boundary conditions to fit the sub-grid
            bound = pp.fvutils.boundary_to_sub_boundary(bound, subcell_topology)
            subface_rhs = False
        # Obtain mappings to exclude boundary faces
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bound, nd)
        # Most of the work is done by submethod for elasticity (which is common for
        # elasticity and poro-elasticity).

        hook, igrad, cell_node_blocks = self._create_inverse_gradient_matrix(
            sd, constit, subcell_topology, bound_exclusion, eta, inverter
        )
        num_sub_cells = cell_node_blocks[0].size

        rhs_cells = self._create_rhs_cell_center(
            sd, subcell_topology, eta, num_sub_cells, bound_exclusion
        )

        hook_igrad = hook * igrad
        # NOTE: This is the point where we expect to reach peak memory need.
        del hook
        # Output should be on face-level (not sub-face)
        hf2f = pp.fvutils.map_hf_2_f(
            subcell_topology.fno_unique, subcell_topology.subfno_unique, nd
        )

        # Stress discretization
        stress = hook_igrad * rhs_cells
        # Right hand side for boundary discretization
        rhs_bound = self._create_bound_rhs(
            bound, bound_exclusion, subcell_topology, sd, subface_rhs
        )
        # Discretization of boundary values
        bound_stress = hook_igrad * rhs_bound

        if not subface_rhs:
            bound_stress = hf2f * bound_stress * hf2f.T
            stress = hf2f * stress

        # Calculate the reconstruction of dispacement at faces
        if hf_eta is None:
            hf_eta = eta
        # We obtain the reconstruction of displacments
        dist_grad, cell_centers = self._reconstruct_displacement(
            sd, subcell_topology, hf_eta
        )

        hf_cell = dist_grad * igrad * rhs_cells + cell_centers
        hf_bound = dist_grad * igrad * rhs_bound

        if not hf_disp:
            # hf2f sums the values, but here we need an average. For now, use simple
            # average, although area weighted values may be more accurate
            num_subfaces = hf2f.sum(axis=1).A.ravel()
            scaling = sps.dia_matrix(
                (1.0 / num_subfaces, 0), shape=(hf2f.shape[0], hf2f.shape[0])
            )

            hf_cell = scaling * hf2f * hf_cell
            hf_bound = scaling * hf2f * hf_bound

        # The subface displacement is given by
        # hf_cell * u_cell_centers + hf_bound * u_bound_condition
        if not subface_rhs:
            hf_bound *= hf2f.T
        return stress, bound_stress, hf_cell, hf_bound

    def _create_inverse_gradient_matrix(
        self,
        sd: pp.Grid,
        constit: pp.FourthOrderTensor,
        subcell_topology: pp.fvutils.SubcellTopology,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
        eta: float,
        inverter: Literal["python", "numba"],
    ) -> tuple[sps.spmatrix, sps.spmatrix, np.ndarray]:
        """
        This is the function where the real discretization takes place. It contains
        the parts that are common for elasticity and poro-elasticity, and was thus
        separated out as a helper function.

        The steps in the discretization are the same as in mpfa (although with
        everything being somewhat more complex since this is a vector equation).
        The mpfa function is currently more clean, so confer that for additional
        comments.

        Parameters:
            sd: Grid
            constit: Constitutive law
            subcell_topology: Wrapper class for numbering of subcell faces, cells
                etc.
            bound_exclusion: Object that can eliminate faces related to boundary
                conditions.
            eta: Parameter determining the continuity point
            inverter: Parameter determining which method to use for inverting the
                local systems

        Returns:
            :obj:`~scipy.sparse.spmatrix`:
                Hooks law, ready to be multiplied with inverse gradients
            :obj:`~scipy.sparse.spmatrix`:
                Inverse gradients

            :obj:`np.ndarray`:
                Relation between cells and vertexes, used to group equations in linear
                system.

        """
        if bound_exclusion.bc_type != "vectorial":
            raise AttributeError("MPSA must be given a vectorial boundary condition")
        nd = sd.dim

        # Compute product between normal vectors and stiffness matrices
        ncsym_all, ncasym, cell_node_blocks, sub_cell_index = self._tensor_vector_prod(
            sd, constit, subcell_topology
        )

        # To avoid singular matrices we are not abe to add the asymetric part of the
        # stress tensor to the Neumann and Robin boundaries for nodes that only has more
        # Neumann-boundary faces than gradients. This will typically happen in the
        # corners where you only can have one gradient for the node. Normally if you
        # have at least one internal face connected to the node you are should be safe.
        # For the Neumann faces we eliminate the asymetic part this does in fact lead to
        # an inconsistency.
        self._eliminate_ncasym_neumann(
            ncasym, subcell_topology, bound_exclusion, cell_node_blocks, nd
        )

        # The final expression of Hook's law will involve deformation gradients on one
        # side of the faces only; eliminate the other one. Note that this must be done
        # before we can pair forces from the two sides of the faces.
        hook = self._unique_hooks_law(ncsym_all, ncasym, subcell_topology, nd)

        # For the Robin boundary conditions we need to pair the forces with the
        # displacement.
        # The contribution of the displacement at the Robin boundary is
        # rob_grad * G + rob_cell * u (this is the displacement at the boundary scaled
        # with the Robin weight times the area of the faces),
        # where G are the gradients and u the cell center displacement. The Robin
        # condtion is then ncsym_rob * G + rob_grad * G + rob_cell * u
        ncsym_full = subcell_topology.pair_over_subfaces_nd(ncsym_all + ncasym)
        del ncasym
        ncsym_rob = bound_exclusion.keep_robin(ncsym_full)
        ncsym_neu = bound_exclusion.keep_neumann(ncsym_full)

        # Book keeping
        num_sub_cells = cell_node_blocks[0].size
        rob_grad, rob_cell = self._get_displacement_submatrices_rob(
            sd, subcell_topology, eta, num_sub_cells, bound_exclusion
        )

        # Pair the forces from each side.
        # ncsym * G is in fact (due to pair_over_subfaces)
        # ncsym_L * G_L + ncsym_R * G_R for the left and right faces.
        # We are here using ncsym_all and not the full tensor
        # ncsym_full = ncsym_all + ncasym. This is because
        # ncasym_L * G_L + ncasym_R * G_R = 0 due to symmetry.
        ncsym = subcell_topology.pair_over_subfaces_nd(ncsym_all)
        del ncsym_all
        # Boundary conditions are taken hand of in ncsym_rob, ncsym_neu or as Dirichlet.
        ncsym = bound_exclusion.exclude_boundary(ncsym)

        # Matrices to enforce displacement continuity
        d_cont_grad, _ = self._get_displacement_submatrices(
            sd, subcell_topology, eta, num_sub_cells, bound_exclusion
        )

        grad_eqs = sps.vstack([ncsym, ncsym_neu, ncsym_rob + rob_grad, d_cont_grad])
        del ncsym, d_cont_grad, ncsym_rob, rob_grad, ncsym_neu

        # To lower the condition number of the local linear systems, the equations that
        # represents stress and displacement continuity, as well as the Robin condition
        # (which is a combination) should ideally have similar scaling. For general
        # grids and material coefficiens, this is not possible to achieve. Nevertheless,
        # we try to achieve reasonably conditioned local problems. A simple approach has
        # turned out to give reasonable results: For all continuity condition (each row)
        # compute the sum of the absolute value of the non-zero elements and scale the
        # entire row with the inverse of the sum. This is equivalent to diagonal
        # left preconditioner for the system. In order not to modify the solution, we
        # will also need to left precondition the right-hand side.
        # The implementation is located in a helper function.
        full_scaling = pp.fvutils.diagonal_scaling_matrix(grad_eqs)

        igrad = (
            self._inverse_gradient(
                full_scaling * grad_eqs,
                sub_cell_index,
                cell_node_blocks,
                subcell_topology.nno_unique,
                bound_exclusion,
                nd,
                inverter,
            )
            * full_scaling
        )

        # Right hand side for cell center variables

        return hook, igrad, cell_node_blocks

    def _create_rhs_cell_center(
        self,
        sd: pp.Grid,
        subcell_topology: pp.fvutils.SubcellTopology,
        eta: float,
        num_sub_cells: int,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
    ) -> sps.spmatrix:
        """Create the right hand side for the cell center discretization.

        Parameters:
            sd: Grid for which the discretization is to be constructed.
            subcell_topology: Data structure for the subcells.
            eta: Location of the displacement continuity point.
            num_sub_cells: Number of subcells in the grid.
            bound_exclusion: Object to exclude boundary conditions.

        Returns:
            Right hand side for the cell center discretization.

        """

        nd = sd.dim

        rob_grad, rob_cell = self._get_displacement_submatrices_rob(
            sd, subcell_topology, eta, num_sub_cells, bound_exclusion
        )

        # The contribution of cell center displacement to stress continuity.
        # This is just zero (T_L + T_R = 0).
        num_subfno = subcell_topology.subfno.max() + 1
        hook_cell = sps.coo_matrix(
            (np.zeros(1), (np.zeros(1), np.zeros(1))),
            shape=(num_subfno * nd, (np.max(subcell_topology.cno) + 1) * nd),
        ).tocsr()
        # Here you have to be carefull if you ever change hook_cell to something else
        # than 0. Because we have pulled the Neumann conditions out of the stress
        # condition the following would give an index error. Instead you would have to
        # make a hook_cell_neu equal the number neumann_sub_faces, and a hook_cell_int
        # equal the number of internal sub_faces and use .keep_neu and .exclude_bnd. But
        # since this is all zeros, this indexing does not matter.
        hook_cell = bound_exclusion.exclude_robin_dirichlet(hook_cell)

        # Matrices to enforce displacement continuity
        _, d_cont_cell = self._get_displacement_submatrices(
            sd, subcell_topology, eta, num_sub_cells, bound_exclusion
        )

        rhs_cells = -sps.vstack([hook_cell, rob_cell, d_cont_cell])

        return rhs_cells

    def _create_bound_rhs(
        self,
        bound: pp.BoundaryConditionVectorial,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
        subcell_topology: pp.fvutils.SubcellTopology,
        sd: pp.Grid,
        subface_rhs: bool,
    ) -> sps.spmatrix:
        """Define rhs matrix to get basis functions for boundary conditions assigned
        face-wise.

        Parameters:
            bound: Boundary condition object.
            bound_exclusion: Object to exclude boundary conditions.
            subcell_topology: Object that carries information on the subcell topology.
            sd: Grid for which the discretization is to be constructed.
            subface_rhs: If True, the right hand side will be defined on subfaces. If
                False, it will be defined on faces.

        Returns:
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.dim * sd.num_faces,
                                                     sd.dim * sd.num_faces))``

            Matrix that can be multiplied with inverse block matrix to get basis
            functions for boundary values. If subface_rhs is True, the matrix will have
            ``sd.dim * subcell_topology.num_subfno_unique`` rows and columns.

        """
        nd = sd.dim

        num_stress = bound_exclusion.exclude_bnd.shape[0]
        num_displ = bound_exclusion.exclude_neu_rob.shape[0]

        num_rob = bound_exclusion.keep_rob.shape[0]
        num_neu = bound_exclusion.keep_neu.shape[0]

        fno = subcell_topology.fno_unique
        subfno = subcell_topology.subfno_unique
        sgn = sd.cell_faces[
            subcell_topology.fno_unique, subcell_topology.cno_unique
        ].A.ravel("F")

        num_dir = np.sum(bound.is_dir)
        if not num_rob == np.sum(bound.is_rob):
            raise AssertionError()
        if not num_neu == np.sum(bound.is_neu):
            raise AssertionError()

        num_bound = num_neu + num_dir + num_rob

        # Obtain the face number for each coordinate
        subfno_nd = np.tile(subfno, (nd, 1)) * nd + np.atleast_2d(np.arange(0, nd)).T

        # Expand the indices Define right hand side for Neumann boundary conditions
        # First row indices in rhs matrix Pick out the subface indices The boundary
        # conditions should be given in the given basis, therefore no transformation
        subfno_neu = bound_exclusion.keep_neumann(
            subfno_nd.ravel("C"), transform=False
        ).ravel("F")
        # Pick out the Neumann boundary
        is_neu_nd = (
            bound_exclusion.keep_neumann(bound.is_neu.ravel("C"), transform=False)
            .ravel("F")
            .astype(bool)
        )

        neu_ind = np.argsort(subfno_neu)
        neu_ind = neu_ind[is_neu_nd[neu_ind]]

        # Robin, same procedure
        subfno_rob = bound_exclusion.keep_robin(
            subfno_nd.ravel("C"), transform=False
        ).ravel("F")

        is_rob_nd = (
            bound_exclusion.keep_robin(bound.is_rob.ravel("C"), transform=False)
            .ravel("F")
            .astype(bool)
        )

        rob_ind = np.argsort(subfno_rob)
        rob_ind = rob_ind[is_rob_nd[rob_ind]]

        # Dirichlet, same procedure
        # remove neumann and robin subfno
        subfno_dir = bound_exclusion.exclude_neumann_robin(
            subfno_nd.ravel("C"), transform=False
        ).ravel("F")
        is_dir_nd = (
            bound_exclusion.exclude_neumann_robin(
                bound.is_dir.ravel("C"), transform=False
            )
            .ravel("F")
            .astype(bool)
        )

        dir_ind = np.argsort(subfno_dir)
        dir_ind = dir_ind[is_dir_nd[dir_ind]]

        # We also need to account for all half faces, that is, do not exclude Dirichlet
        # and Neumann boundaries. This is the global indexing.
        is_neu_all = bound.is_neu.ravel("C")
        neu_ind_all = np.argwhere(
            np.reshape(is_neu_all, (nd, -1), order="C").ravel("F")
        ).ravel("F")
        is_dir_all = bound.is_dir.ravel("C")
        dir_ind_all = np.argwhere(
            np.reshape(is_dir_all, (nd, -1), order="C").ravel("F")
        ).ravel("F")

        is_rob_all = bound.is_rob.ravel("C")
        rob_ind_all = np.argwhere(
            np.reshape(is_rob_all, (nd, -1), order="C").ravel("F")
        ).ravel("F")

        # We now merge the neuman and robin indices since they are treated equivalent.
        # Remember that the first set of local equations are the stress equilibrium for
        # the internall faces. The number of internal stresses therefore has to be added
        # to the Neumann and Robin indices.
        if rob_ind.size == 0:
            neu_rob_ind = neu_ind + num_stress
        elif neu_ind.size == 0:
            neu_rob_ind = rob_ind + num_stress
        else:
            neu_rob_ind = np.hstack(
                (neu_ind + num_stress, rob_ind + num_stress + num_neu)
            )

        neu_rob_ind_all = np.hstack((neu_ind_all, rob_ind_all))

        # stack together
        bnd_ind = np.hstack((neu_rob_ind_all, dir_ind_all))

        # Some care is needed to compute coefficients in Neumann matrix: sgn is already
        # defined according to the subcell topology [fno], while areas must be drawn
        # from the grid structure, and thus go through fno
        fno_ext = np.tile(fno, nd)
        num_face_nodes = sd.face_nodes.sum(axis=0).A.ravel("F")

        # Coefficients in the matrix. For the Neumann boundary components we set the
        # value as seen from the outside of the domain. Note that they do not have to do
        # so, and we will flip the sign later. This means that a stress [1,1] on a
        # boundary face pushes(or pulls) the face to the top right corner.
        if subface_rhs:
            # In this case we set the rhs for the sub-faces. Note that the rhs values
            # should be integrated over the subfaces, that is
            # stress_neumann *\cdot * normal * subface_area
            neu_val = 1 * np.ones(neu_rob_ind_all.size)
        else:
            # In this case we set the value at a face, thus, we need to distribute the
            #  face values to the subfaces. We do this by an area-weighted average. Note
            # that the rhs values should in this case be integrated over the faces, that
            # is: stress_neumann *\cdot * normal * face_area
            neu_val = 1 / num_face_nodes[fno_ext[neu_rob_ind_all]]

        # The columns will be 0:neu_rob_ind.size
        if neu_rob_ind.size > 0:
            neu_cell = sps.coo_matrix(
                (neu_val.ravel("F"), (neu_rob_ind, np.arange(neu_rob_ind.size))),
                shape=(num_stress + num_neu + num_rob, num_bound),
            ).tocsr()
        else:
            # Special handling when no elements are found. Not sure if this is
            # necessary, or if it is me being stupid
            neu_cell = sps.coo_matrix((num_stress + num_rob, num_bound)).tocsr()

        # For Dirichlet, the coefficients in the matrix should be duplicated the same
        # way as the row indices, but with no increment
        sgn_nd = np.tile(sgn, (nd, 1)).ravel("F")
        dir_val = sgn_nd[dir_ind_all]
        del sgn_nd
        # Column numbering starts right after the last Neumann column. dir_val is
        # ordered [u_x_1, u_y_1, u_x_2, u_y_2, ...], and dir_ind shuffles this ordering.
        # The final matrix will first have the x-coponent of the displacement for each
        # face, then the y-component, etc.
        if dir_ind.size > 0:
            dir_cell = sps.coo_matrix(
                (dir_val, (dir_ind, num_neu + num_rob + np.arange(dir_ind.size))),
                shape=(num_displ, num_bound),
            ).tocsr()
        else:
            # Special handling when no elements are found. Not sure if this is
            # necessary, or if it is me being stupid
            dir_cell = sps.coo_matrix((num_displ, num_bound)).tocsr()

        num_subfno = np.max(subfno) + 1

        # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1. Map these
        # to all half-face indices
        bnd_2_all_hf = sps.coo_matrix(
            (np.ones(num_bound), (np.arange(num_bound), bnd_ind)),
            shape=(num_bound, num_subfno * nd),
        )

        # the rows of rhs_bound will be ordered with first the x-component of all
        # neumann faces, then the y-component of all Neumann faces, then the z-component
        # of all Neumann faces. Then we will have the equivalent for the Dirichlet
        # faces.

        rhs_bound = sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf

        return rhs_bound

    def _reconstruct_displacement(
        self,
        sd: pp.Grid,
        subcell_topology: pp.fvutils.SubcellTopology,
        eta: Optional[float] = None,
    ) -> tuple[sps.csr_matrix, sps.csr_matrix]:
        """Function for reconstructing the displacement at the half faces given the
        local gradients.

        For a subcell Ks associated with cell K and node s, the displacement at a point
        x is given by
            U_Ks + G_Ks (x - x_k),
        x_K is the cell center of cell k. The point at which we evaluate the displacement
        is given by eta, which is equivalent to the continuity points in mpsa.
        For an internal subface we will obtain two values for the displacement,
        one for each of the cells associated with the subface. The displacement given
        here is the average of the two. Note that at the continuity points the two
        displacements will by construction be equal.

        Parameters:
            sd: Grid
            subcell_topology: Wrapper class for numbering of subcell faces, cells
                etc.
            eta (float range=[0,1)): Optional. Parameter determining the
                point at which the displacement is evaluated. If ``eta`` is not given
                the method will call ``:meth:~pp.numerics.fv.fvutils.determine_eta`` to
                set it.

        Returns:
            ``scipy.sparse.csr_matrix (sd.dim*num_sub_faces, sd.dim*num_cells)``:


                Reconstruction matrix for the displacement at the half faces. This is
                the contribution from the cell-center displacements. The half-face
                displacements are ordered sub-face_wise, i.e.,
                        ``(U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)``

            ``scipy.sparse.csr_matrix (sd.dim*num_sub_faces, sd.dim*num_faces)``:

                Reconstruction matrix for the displacement at the half faces. This is
                the contribution from the boundary conditions. The half-face
                displacements are ordered sub_face wise, i.e.,
                        ``(U_x_0, U_x_1, ..., U_x_n, U_y0, U_y1, ...)``

        """
        if eta is None:
            eta = pp.fvutils.determine_eta(sd)

        # Calculate the distance from the cell centers to continuity points
        D_g = pp.fvutils.compute_dist_face_cell(
            sd, subcell_topology, eta, return_paired=False
        )
        # We here average the contribution on internal sub-faces. If you want to get out
        # both displacements on a sub-face your can remove the averaging.
        _, IC, counts = np.unique(
            subcell_topology.subfno, return_inverse=True, return_counts=True
        )

        avg_over_subfaces = sps.coo_matrix(
            (1 / counts[IC], (subcell_topology.subfno, subcell_topology.subhfno))
        )
        D_g = avg_over_subfaces * D_g
        # expand indices to x-y-z
        D_g = sps.kron(sps.eye(sd.dim), D_g)
        D_g = D_g.tocsr()

        # Get a mapping from cell centers to half-faces
        D_c = sps.coo_matrix(
            (1 / counts[IC], (subcell_topology.subfno, subcell_topology.cno))
        ).tocsr()
        # Expand indices to x-y-z
        D_c = sps.kron(sps.eye(sd.dim), D_c)
        D_c = D_c.tocsc()
        # book keeping
        cell_node_blocks, _ = pp.matrix_operations.rlencode(
            np.vstack((subcell_topology.cno, subcell_topology.nno))
        )
        num_sub_cells = cell_node_blocks[0].size
        # The column ordering of the displacement equilibrium equations are formed as a
        # Kronecker product of scalar equations. Bring them to the same form as that
        # applied in the force balance equations
        dist_grad, cell_centers = self._rearange_columns_displacement_eqs(
            D_g, D_c, num_sub_cells, sd.dim
        )
        # The row ordering is now first variable x of all subfaces then variable y of
        # all subfaces, etc. Change the ordering to first all variables of first cell,
        # then all variables of second cell, etc.
        P = self._row_major_to_col_major(cell_centers.shape, sd.dim, 0)
        return P * dist_grad, P * cell_centers

    # -----------------------------------------------------------------------------
    #
    # Below here are helper functions, which tend to be less than well documented.
    #
    # -----------------------------------------------------------------------------

    def _estimate_peak_memory_mpsa(self, sd: pp.Grid) -> int:
        """Compute a rough estimate of peak memory need for the discretization.

        Parameters:
            sd: The grid to be discretized

        Returns:
            The estimated peak memory need.

        """
        nd = sd.dim
        num_cell_nodes = sd.cell_nodes().sum(axis=1).A

        # Number of unknowns around a vertex: nd^2 per cell that share the vertex for
        # pressure gradients, and one per cell (cell center pressure)
        num_grad_unknowns = nd**2 * num_cell_nodes

        # The most expensive field is the storage of igrad, which is block diagonal with
        # num_grad_unknowns sized blocks. The number of elements is the square of the
        # local system size. The factor 2 accounts for matrix storage in sparse format
        # (rows and data; ignore columns since this is in compressed format)
        igrad_size = np.power(num_grad_unknowns, 2).sum() * 2

        # The discretization of Hook's law will require nd^2 (that is, a gradient) per
        # sub-face per dimension
        num_sub_face = sd.face_nodes.sum()
        hook_size = nd * num_sub_face * nd**2

        # Balancing of stresses will require 2*nd**2 (gradient on both sides) fields per
        # sub-face per dimension
        nk_grad_size = 2 * nd * num_sub_face * nd**2
        # Similarly, pressure continuity requires 2 * (nd+1) (gradient on both sides,
        # and cell center pressures) numbers
        pr_cont_size = 2 * (nd**2 + 1) * num_sub_face * nd

        total_size = igrad_size + hook_size + nk_grad_size + pr_cont_size

        # Not covered yet is various fields on subcell topology, mapping matrices
        # between local and block ordering etc.
        return total_size

    def _get_displacement_submatrices(
        self,
        sd: pp.Grid,
        subcell_topology: pp.fvutils.SubcellTopology,
        eta: float,
        num_sub_cells: int,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
    ) -> tuple[sps.spmatrix, sps.spmatrix]:
        """Get the submatrices for the displacement balance equations.

        Parameters:
            sd: The grid to be discretized
            subcell_topology: A class containing information on the subcell topology.
            eta: Describes the location of the displacement conctinuity points.
            num_sub_cells: Number of sub-cells in the grid.
            bound_exclusion: A class to exclude boundary conditions from the local
                systems.

        Returns:
            A tuple of two matrices:

            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.dim * num_half_faces,
                                                     sd.dim * sd.num_cells))``

            Contribution from cell center displacements to continuity equations.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.dim * num_half_faces,
                                                     sd.dim * sd.num_faces))``

            Contribution from boundary conditions to continuity equations.

        """
        nd = sd.dim
        # Distance from cell centers to face centers, this will be the contribution from
        # gradient unknown to equations for displacement continuity.
        d_cont_grad = pp.fvutils.compute_dist_face_cell(sd, subcell_topology, eta)

        # For force balance, displacements and stresses on the two sides of the
        # matrices must be paired.
        d_cont_grad = sps.kron(sps.eye(nd), d_cont_grad)

        # Contribution from cell center potentials to local systems
        d_cont_cell = self._cell_variable_contribution(sd, subcell_topology)

        # Expand equations for displacement balance, and eliminate rows associated with
        # neumann boundary conditions
        d_cont_grad = bound_exclusion.exclude_neumann_robin(d_cont_grad)
        d_cont_cell = bound_exclusion.exclude_neumann_robin(d_cont_cell)

        # The column ordering of the displacement equilibrium equations are formed as a
        # Kronecker product of scalar equations. Bring them to the same form as that
        # applied in the force balance equations
        d_cont_grad, d_cont_cell = self._rearange_columns_displacement_eqs(
            d_cont_grad, d_cont_cell, num_sub_cells, nd
        )

        return d_cont_grad, d_cont_cell

    def _get_displacement_submatrices_rob(
        self,
        sd: pp.Grid,
        subcell_topology: pp.fvutils.SubcellTopology,
        eta: float,
        num_sub_cells: int,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
    ) -> tuple[sps.spmatrix, sps.spmatrix]:
        """Get the submatrices for the displacement balance equations for Robin
        conditions.

        Parameters:
            sd: The grid to be discretized
            subcell_topology: A class containing information on the subcell topology.
            eta: Describes the location of the displacement conctinuity points.
            num_sub_cells: Number of sub-cells in the grid.
            bound_exclusion: A class to exclude boundary conditions from the local
                systems.

        Returns:
            A tuple of two matrices:

            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.dim * num_half_faces,
                                                     sd.dim * sd.num_cells))``

            Contribution from cell center displacements to continuity equations.
            :obj:`~scipy.sparse.spmatrix`: ``(shape=(sd.dim * num_half_faces,
                                                     sd.dim * sd.num_faces))``

            Contribution from boundary conditions to continuity equations.

        """
        nd = sd.dim
        # Distance from cell centers to face centers, this will be the
        # contribution from gradient unknown to equations for displacement
        # at the boundary
        rob_grad = pp.fvutils.compute_dist_face_cell(sd, subcell_topology, eta)

        # For the Robin condition the distance from the cell centers to face centers
        # will be the contribution from the gradients. We integrate over the subface
        # and multiply by the area
        num_nodes = np.diff(sd.face_nodes.indptr)
        sgn = sd.cell_faces[subcell_topology.fno_unique, subcell_topology.cno_unique].A
        scaled_sgn = (
            sgn[0]
            * sd.face_areas[subcell_topology.fno_unique]
            / num_nodes[subcell_topology.fno_unique]
        )
        # pair_over_subfaces flips the sign so we flip it back
        rob_grad = sps.kron(sps.eye(nd), sps.diags(scaled_sgn) * rob_grad)
        # Contribution from cell center potentials to local systems
        rob_cell = sps.coo_matrix(
            (
                sd.face_areas[subcell_topology.fno] / num_nodes[subcell_topology.fno],
                (subcell_topology.subfno, subcell_topology.cno),
            )
        ).tocsr()
        rob_cell = sps.kron(sps.eye(nd), rob_cell)

        # First do a basis transformation
        rob_grad = bound_exclusion.basis_matrix * rob_grad
        rob_cell = bound_exclusion.basis_matrix * rob_cell
        # And apply the robin weight in the rotated basis
        rob_grad = bound_exclusion.robin_weight * rob_grad
        rob_cell = bound_exclusion.robin_weight * rob_cell
        # Expand equations for displacement balance, and keep rows associated with
        # neumann boundary conditions. Remember we have already rotated the basis above
        rob_grad = bound_exclusion.keep_robin(rob_grad, transform=False)
        rob_cell = bound_exclusion.keep_robin(rob_cell, transform=False)

        # The column ordering of the displacement equilibrium equations are formed as a
        # Kronecker product of scalar equations. Bring them to the same form as that
        # applied in the force balance equations
        rob_grad, rob_cell = self._rearange_columns_displacement_eqs(
            rob_grad, rob_cell, num_sub_cells, nd
        )
        return rob_grad, rob_cell

    def _split_stiffness_matrix(
        self, constit: pp.FourthOrderTensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split the stiffness matrix into symmetric and asymetric part

        Parameters:
            constit: The stiffness matrix.

        Returns:
            The symmetric part of the stiffness matrix.

        The asymmetric part of the stiffness matrix.

        """
        dim = np.sqrt(constit.values.shape[0])

        # We do not know how constit is used outside the discretization, so create deep
        # copies to avoid overwriting. Not really sure if this is necessary
        csym = 0 * constit.copy().values
        casym = constit.copy().values

        # The copy constructor for the stiffness matrix will represent all dimensions as
        # 3d. If dim==2, delete the redundant rows and columns
        if dim == 2 and csym.shape[0] == 9:
            csym = np.delete(csym, (2, 5, 6, 7, 8), axis=0)
            csym = np.delete(csym, (2, 5, 6, 7, 8), axis=1)
            casym = np.delete(casym, (2, 5, 6, 7, 8), axis=0)
            casym = np.delete(casym, (2, 5, 6, 7, 8), axis=1)

        # The splitting is hard coded based on the ordering of elements in the stiffness
        # matrix
        if dim == 2:
            csym[0, 0] = casym[0, 0]
            csym[1, 1] = casym[1, 1]
            csym[2, 2] = casym[2, 2]
            csym[3, 0] = casym[3, 0]
            csym[0, 3] = casym[0, 3]
            csym[3, 3] = casym[3, 3]
        else:  # dim == 3
            csym[0, 0] = casym[0, 0]
            csym[1, 1] = casym[1, 1]
            csym[2, 2] = casym[2, 2]
            csym[3, 3] = casym[3, 3]
            csym[4, 4] = casym[4, 4]
            csym[5, 5] = casym[5, 5]
            csym[6, 6] = casym[6, 6]
            csym[7, 7] = casym[7, 7]
            csym[8, 8] = casym[8, 8]

            csym[4, 0] = casym[4, 0]
            csym[8, 0] = casym[8, 0]
            csym[0, 4] = casym[0, 4]
            csym[8, 4] = casym[8, 4]
            csym[0, 8] = casym[0, 8]
            csym[4, 8] = casym[4, 8]
        # The asymmetric part is whatever is not in the symmetric part
        casym -= csym
        return csym, casym

    def _tensor_vector_prod(
        self,
        sd: pp.Grid,
        constit: pp.FourthOrderTensor,
        subcell_topology: pp.fvutils.SubcellTopology,
    ) -> tuple[sps.spmatrix, sps.spmatrix, np.ndarray, np.ndarray]:
        """Compute product between stiffness tensor and face normals.

        The method splits the stiffness matrix into a symmetric and asymmetric
        part, and computes the products with normal vectors for each. The method
        also provides a unique identification of sub-cells (in the form of pairs of
        cells and nodes), and a global numbering of subcell gradients.

        Parameters:
            sd: grid
            constit: Stiffness matrix, in the form of a fourth order tensor.
            subcell_topology: Numberings of subcell quantities etc.

        Returns:
            ncsym, ncasym: Product with face normals for symmetric and asymmetric part
            of stiffness tensors. On the subcell level. In effect, these will be
            stresses on subfaces, as functions of the subcell gradients (to be computed
            somewhere else). The rows first represent stresses in the x-direction for
            all faces, then y direction etc.

            Unique pairing of cell and node numbers for subcells, as a 2xn array. The
            first row gives cell numbers, the second node numbers. Together, this cell-
            node combination identifies a unique subcell.

            Numbering scheme for subcell gradients - gives a global numbering for the
            gradients. One column per subcell, the rows gives the index for the
            individual components of the gradients.

        """

        # Stack cells and nodes, and remove duplicate rows. Since subcell_mapping
        # defines cno and nno (and others) working cell-wise, this will correspond to a
        # unique rows (Matlab-style) from what I understand. This also means that the
        # pairs in cell_node_blocks uniquely defines subcells, and can be used to index
        # gradients etc.
        cell_node_blocks, blocksz = pp.matrix_operations.rlencode(
            np.vstack((subcell_topology.cno, subcell_topology.nno))
        )

        nd = sd.dim

        # Duplicates in [cno, nno] corresponds to different faces meeting at the same
        # node. There should be exactly nd of these. This test will fail for pyramids in
        # 3D
        if not np.all(blocksz == nd):
            raise AssertionError()

        # Define row and column indices to be used for normal vector matrix Rows are
        # based on sub-face numbers. Columns have nd elements for each sub-cell (to
        # store a vector) and is adjusted according to block sizes
        _, cn = np.meshgrid(subcell_topology.subhfno, np.arange(nd))
        sum_blocksz = np.cumsum(blocksz)
        cn += pp.matrix_operations.rldecode(sum_blocksz - blocksz[0], blocksz)
        ind_ptr_n = np.hstack((np.arange(0, cn.size, nd), cn.size))

        # Distribute faces equally on the sub-faces, and store in a matrix
        num_nodes = np.diff(sd.face_nodes.indptr)
        normals = (
            sd.face_normals[:, subcell_topology.fno] / num_nodes[subcell_topology.fno]
        )
        normals_mat = sps.csr_matrix((normals.ravel("F"), cn.ravel("F"), ind_ptr_n))

        # Then row and columns for stiffness matrix. There are nd^2 elements in the
        # gradient operator, and so the structure is somewhat different from the normal
        # vectors
        _, cc = np.meshgrid(subcell_topology.subhfno, np.arange(nd**2))
        sum_blocksz = np.cumsum(blocksz**2)
        cc += pp.matrix_operations.rldecode(sum_blocksz - blocksz[0] ** 2, blocksz)
        ind_ptr_c = np.hstack((np.arange(0, cc.size, nd**2), cc.size))

        # Splitt stiffness matrix into symmetric and anti-symmatric part
        sym_tensor, asym_tensor = self._split_stiffness_matrix(constit)

        # Getting the right elements out of the constitutive laws was a bit tricky, but
        # the following code turned out to do the trick
        sym_tensor_swp = np.swapaxes(sym_tensor, 2, 0)
        asym_tensor_swp = np.swapaxes(asym_tensor, 2, 0)

        # The first dimension in csym and casym represent the contribution from all
        # dimensions to the stress in one dimension (in 2D, csym[0:2,:, :] together
        # gives stress in the x-direction etc.
        # Define index vector to access the right rows
        rind = np.arange(nd)

        # Empty matrices to initialize matrix-tensor products. Will be expanded
        # as we move on
        zr = np.zeros(0)
        ncsym = sps.coo_matrix((zr, (zr, zr)), shape=(0, cc.max() + 1)).tocsr()
        ncasym = sps.coo_matrix((zr, (zr, zr)), shape=(0, cc.max() + 1)).tocsr()

        # For the asymmetric part of the tensor, we will apply volume averaging.
        # Associate a volume with each sub-cell, and a node-volume as the sum of all
        # surrounding sub-cells.
        num_cell_nodes = sd.num_cell_nodes()
        cell_vol = sd.cell_volumes / num_cell_nodes
        node_vol = (
            np.bincount(subcell_topology.nno, weights=cell_vol[subcell_topology.cno])
            / sd.dim
        )

        num_elem = cell_node_blocks.shape[1]
        map_mat = sps.coo_matrix(
            (np.ones(num_elem), (np.arange(num_elem), cell_node_blocks[1]))
        )
        weight_mat = sps.coo_matrix(
            (
                cell_vol[cell_node_blocks[0]] / node_vol[cell_node_blocks[1]],
                (cell_node_blocks[1], np.arange(num_elem)),
            )
        )
        # Operator for carying out the average
        average = sps.kron(map_mat * weight_mat, sps.identity(nd)).tocsr()

        for iter1 in range(nd):
            # Pick out part of Hook's law associated with this dimension The code here
            # looks nasty, it should be possible to get the right format of the
            # submatrices in a simpler way, but I couldn't do it.
            sym_dim = np.concatenate(sym_tensor_swp[:, :, rind], axis=1).transpose()
            asym_dim = np.concatenate(asym_tensor_swp[:, :, rind], axis=1).transpose()

            # Distribute (relevant parts of) Hook's law on subcells This will be nd
            # rows, thus cell ci is associated with indices ci*nd+np.arange(nd)
            sub_cell_ind = pp.fvutils.expand_indices_nd(cell_node_blocks[0], nd)
            sym_vals = sym_dim[sub_cell_ind]
            asym_vals = asym_dim[sub_cell_ind]

            # Represent this part of the stiffness matrix in matrix form
            csym_mat = sps.csr_matrix((sym_vals.ravel("C"), cc.ravel("F"), ind_ptr_c))
            casym_mat = sps.csr_matrix((asym_vals.ravel("C"), cc.ravel("F"), ind_ptr_c))

            # Compute average around vertexes
            casym_mat = average * casym_mat

            # Compute products of normal vectors and stiffness tensors, and stack
            # dimensions vertically
            ncsym = sps.vstack((ncsym, normals_mat * csym_mat))
            ncasym = sps.vstack((ncasym, normals_mat * casym_mat))

            # Increase index vector, so that we get rows contributing to forces
            # in the next dimension
            rind += nd

        grad_ind = cc[:, ::nd]

        return ncsym, ncasym, cell_node_blocks, grad_ind

    def _inverse_gradient(
        self,
        grad_eqs: sps.spmatrix,
        sub_cell_index: np.ndarray,
        cell_node_blocks: np.ndarray,
        nno_unique: np.ndarray,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
        nd: int,
        inverter: str,
    ) -> sps.spmatrix:
        """Invert local system to compute the subcell gradient operator.

        Parameters:
            grad_eqs: Local system to be inverted.
            sub_cell_index: Index of all subcells.
            cell_node_blocks: Pairs of cell and node pairs, which defines sub-cells.
            nno_unique: Unique nodes in the grid.
            bound_exclusion: Object to exclude boundary conditions.
            nd: Number of dimensions.
            inverter: Inverter to use.

        Returns:
            Inverse of the local system.

        """

        # Mappings to convert linear system to block diagonal form
        rows2blk_diag, cols2blk_diag, size_of_blocks = self._block_diagonal_structure(
            sub_cell_index, cell_node_blocks, nno_unique, bound_exclusion, nd
        )

        grad = rows2blk_diag * grad_eqs * cols2blk_diag
        # Compute inverse gradient operator, and map back again
        igrad = (
            cols2blk_diag
            * pp.matrix_operations.invert_diagonal_blocks(
                grad, size_of_blocks, method=inverter
            )
            * rows2blk_diag
        )
        logger.debug("max igrad: " + str(np.max(np.abs(igrad))))
        return igrad

    def _block_diagonal_structure(
        self,
        sub_cell_index: np.ndarray,
        cell_node_blocks: np.ndarray,
        nno: np.ndarray,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
        nd: int,
    ) -> tuple[sps.spmatrix, sps.spmatrix, np.ndarray]:
        """Define matrices to turn linear system into block-diagonal form.

        Parameters
            sub_cell_index: Index of all subcells.
            cell_node_blocks: Pairs of cell and node pairs, which defines sub-cells.
            nno: Node numbers associated with balance equations
            bound_exclusion: Object to eliminate faces associated with different types
                of boundary conditions.
            nd: Number of spatial dimensions.

        Returns:
            :obj:`~scipy.sparse.spmatrix`: Transform rows of linear system to
                block-diagonal form.

            :obj:`~scipy.sparse.spmatrix`: Transform colmns of linear system to
                block-diagonal form.

            np.ndarray: Number of equations in each block.

        """
        # Stack node numbers of equations on top of each other, and sort them to get
        # block-structure. First eliminate node numbers at the boundary, where the
        # equations are either of flux or pressure continuity (not both).

        nno = np.tile(nno, nd)
        # The node numbers do not have a basis, so no transformation here. We just
        # want to pick out the correct node numbers for the correct equations.
        nno_stress = bound_exclusion.exclude_boundary(nno, transform=False)
        nno_displacement = bound_exclusion.exclude_neumann_robin(nno, transform=False)
        nno_neu = bound_exclusion.keep_neumann(nno, transform=False)
        nno_rob = bound_exclusion.keep_robin(nno, transform=False)
        node_occ = np.hstack((nno_stress, nno_neu, nno_rob, nno_displacement))

        sorted_ind = np.argsort(node_occ, kind="mergesort")
        rows2blk_diag = sps.coo_matrix(
            (np.ones(sorted_ind.size), (np.arange(sorted_ind.size), sorted_ind))
        ).tocsr()
        # Size of block systems
        sorted_nodes_rows = node_occ[sorted_ind]
        size_of_blocks = np.bincount(sorted_nodes_rows.astype("int64"))

        # cell_node_blocks[1] contains the node numbers associated with each sub-cell
        # gradient (and so column of the local linear systems). A sort of these will
        # give a block-diagonal structure
        sorted_nodes_cols = np.argsort(cell_node_blocks[1], kind="mergesort")
        subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel("F")
        cols2blk_diag = sps.coo_matrix(
            (
                np.ones(sub_cell_index.size),
                (subcind_nodes, np.arange(sub_cell_index.size)),
            )
        ).tocsr()
        return rows2blk_diag, cols2blk_diag, size_of_blocks

    def _unique_hooks_law(
        self,
        csym: np.ndarray,
        casym: np.ndarray,
        subcell_topology: pp.fvutils.SubcellTopology,
        nd: int,
    ) -> sps.spmatrix:
        """Obtain discrete Hook's law for the sub-cell gradients.


        Go from products of normal vectors with stiffness matrices (symmetric and
        asymmetric), covering both sides of faces, to a discrete Hook's law, that, when
        multiplied with sub-cell gradients, will give face stresses

        Parameters:
            csym: Symmetric part of stiffness matrix.
            casym: Asymmetric part of stiffness matrix.
            subcell_topology: Object containing information about sub-cells.
            nd: Number of spatial dimensions.

        Returns:
            Discrete Hook's law.

        """
        # unique_sub_fno covers scalar equations only. Extend indices to cover multiple
        # dimensions
        num_eqs = csym.shape[0] / nd
        ind_single = np.tile(subcell_topology.unique_subfno, (nd, 1))
        increments = np.arange(nd) * num_eqs
        ind_all = np.reshape(ind_single + increments[:, np.newaxis], -1).astype(int)

        # Unique part of symmetric and asymmetric products
        hook_sym = csym[ind_all, ::]
        hook_asym = casym[ind_all, ::]

        # Hook's law, as it comes out of the normal-vector * stiffness matrix is sorted
        # with x-component balances first, then y-, etc. Sort this to a face-wise
        # ordering
        comp2face_ind = np.argsort(
            np.tile(subcell_topology.subfno_unique, nd), kind="mergesort"
        )
        comp2face = sps.coo_matrix(
            (
                np.ones(comp2face_ind.size),
                (np.arange(comp2face_ind.size), comp2face_ind),
            ),
            shape=(comp2face_ind.size, comp2face_ind.size),
        )
        hook = comp2face * (hook_sym + hook_asym)

        return hook

    def _cell_variable_contribution(
        self, sd: pp.Grid, subcell_topology: pp.fvutils.SubcellTopology
    ) -> sps.spmatrix:
        """Construct contribution from cell center variables to local systems.

        For stress equations, these are zero, while for cell centers it is +- 1
        Parameters:
            sd: Grid to discretize.
            subcell_topology: Object containing information about sub-cells.

        Returns:
            Contribution from cell center variables to local systems.

        """
        nd = sd.dim
        sgn = sd.cell_faces[subcell_topology.fno, subcell_topology.cno].A

        # Contribution from cell center potentials to local systems
        # For pressure continuity, +-1
        d_cont_cell = sps.coo_matrix(
            (sgn[0], (subcell_topology.subfno, subcell_topology.cno))
        ).tocsr()
        d_cont_cell = sps.kron(sps.eye(nd), d_cont_cell)
        # Zero contribution to stress continuity

        return d_cont_cell

    def _rearange_columns_displacement_eqs(
        self,
        d_cont_grad: np.ndarray,
        d_cont_cell: np.ndarray,
        num_sub_cells: int,
        nd: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform columns of displacement balance from increasing cell
        ordering (first x-variables of all cells, then y) to increasing variables (first
        all variables of the first cells, then...)

        Parameters:
            d_cont_grad: Contribution from sub-cell gradients to local systems.
            d_cont_cell: Contribution from cell center variables to local systems.
            num_sub_cells: Number of sub-cells.
            nd: Number of spatial dimensions.

        Returns:
            tuple[np.ndarray, np.ndarray]: Rearranged contributions from sub-cell
            gradients and cell center variables.

        """
        # Repeat sub-cell indices nd times. Fortran ordering (column major) gives same
        # ordering of indices as used for the scalar equation (where there are nd
        # gradient variables for each sub-cell), and thus the format of each block in
        # d_cont_grad
        rep_ci_single_blk = np.tile(np.arange(num_sub_cells), (nd, 1)).reshape(
            -1, order="F"
        )
        # Then repeat the single-block indices nd times (corresponding to the way
        # d_cont_grad is constructed by Kronecker product), and find the sorting indices
        d_cont_grad_map = np.argsort(np.tile(rep_ci_single_blk, nd), kind="mergesort")
        # Use sorting indices to bring d_cont_grad to the same order as that used for
        # the columns in the stress continuity equations
        d_cont_grad = d_cont_grad[:, d_cont_grad_map]
        # For the cell displacement variables, we only need a single expansion (
        # corresponding to the second step for the gradient unknowns)
        num_cells = d_cont_cell.shape[1] / nd
        d_cont_cell_map = np.argsort(
            np.tile(np.arange(num_cells), nd), kind="mergesort"
        )
        d_cont_cell = d_cont_cell[:, d_cont_cell_map]
        return d_cont_grad, d_cont_cell

    def _row_major_to_col_major(self, shape: tuple, nd: int, axis: int) -> sps.spmatrix:
        """Transform columns of displacement balance from increasing cell
        ordering (first x-variables of all cells, then y) to increasing variables (first
        all variables of the first cells, then...)

        Parameters:
            shape: Shape of the matrix to be transformed.
            nd: Number of spatial dimensions.
            axis: Axis along which the transformation should be performed.

        Returns:
            Rearranged matrix.

        """
        P = sps.diags(np.ones(shape[axis])).tocsr()
        num_var = shape[axis] / nd
        mapping = np.argsort(np.tile(np.arange(num_var), nd), kind="mergesort")
        if axis == 1:
            P = P[:, mapping]
        elif axis == 0:
            P = P[mapping, :]
        else:
            raise ValueError("axis must be 0 or 1")
        return P

    def _eliminate_ncasym_neumann(
        self,
        ncasym: np.ndarray,
        subcell_topology: pp.fvutils.SubcellTopology,
        bound_exclusion: pp.fvutils.ExcludeBoundaries,
        cell_node_blocks: np.ndarray,
        nd: int,
    ) -> None:
        """Eliminate the asymetric part of the stress tensor such that the local systems
        are invertible.

        Parameters:
            ncasym: Asymetric part of the stress tensor.
            subcell_topology: Object containing information about sub-cells.
            bound_exclusion: Object containing information about excluded boundaries.
            cell_node_blocks: Pairs of node and cell indices that identify sub-cells.
            nd: Number of spatial dimensions.

        """
        # We expand the node indices such that we get one indices for each vector
        # equation. The equations are ordered as first all x, then all y, and so on
        node_blocks_nd = np.tile(cell_node_blocks[1], (nd, 1))
        node_blocks_nd += subcell_topology.num_nodes * np.atleast_2d(np.arange(0, nd)).T
        nno_nd = np.tile(subcell_topology.nno_unique, (nd, 1))
        nno_nd += subcell_topology.num_nodes * np.atleast_2d(np.arange(0, nd)).T

        # Each local system is associated to a node. We count the number of subcells for
        # assoiated with each node.
        _, num_sub_cells = np.unique(node_blocks_nd.ravel("C"), return_counts=True)

        # Then we count the number how many Neumann subfaces there are for each node.
        nno_neu = bound_exclusion.keep_neumann(nno_nd.ravel("C"), transform=False)
        _, idx_neu, count_neu = np.unique(
            nno_neu, return_inverse=True, return_counts=True
        )

        # The local system is invertible if the number of sub_cells (remember there is
        # one gradient for each subcell) is larger than the number of Neumann sub_faces.
        # To obtain an invertible system we remove the asymetric part around these
        # nodes.
        count_neu = bound_exclusion.keep_neu.T * count_neu[idx_neu]
        diff_count = num_sub_cells[nno_nd.ravel("C")] - count_neu
        remove_singular = np.argwhere((diff_count < 0)).ravel()

        # remove_singular gives the indices of the subfaces. We now obtain the indices
        # as given in ncasym,
        subfno_nd = np.tile(subcell_topology.unique_subfno, (nd, 1))
        subfno_nd += subcell_topology.fno.size * np.atleast_2d(np.arange(0, nd)).T
        dof_elim = subfno_nd.ravel("C")[remove_singular]
        # and eliminate the rows corresponding to these subfaces
        pp.matrix_operations.zero_rows(ncasym, dof_elim)
        logger.debug("number of ncasym eliminated: " + str(np.sum(dof_elim.size)))

    def _reduce_grid_constit_2d(
        self, sd: pp.Grid, constit: pp.FourthOrderTensor
    ) -> tuple[pp.Grid, pp.FourthOrderTensor]:
        """Reduce a constitutive law written for a 3d grid to a 2d grid.

        Parameters:
            sd: Grid to be discretized.
            constit: Constitutive law for the 3d grid.

        Returns:
            A tuple with the following elements:

                Reduced grid.

                Reduced constitutive law.

        """
        sd = sd.copy()

        (
            cell_centers,
            face_normals,
            face_centers,
            _,
            _,
            nodes,
        ) = pp.map_geometry.map_grid(sd)
        sd.cell_centers = cell_centers
        sd.face_normals = face_normals
        sd.face_centers = face_centers
        sd.nodes = nodes

        # The stiffness matrix should also be rotated before deleting rows and columns.
        # However, for isotropic media, the standard __init__ for the FourthOrderTensor,
        # followed by the below deletions will in effect generate just what we wanted
        # (assuming we are happy with the Lame parameters, and do not worry about
        # plane-strain / plane-stress consistency). That is all to say, this is a bit
        # inconsistent, but it may just end up okay.
        constit = constit.copy()
        constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=0)
        constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=1)
        return sd, constit

    def _bc_for_subgrid(
        self, bc: pp.BoundaryConditionVectorial, sub_g: pp.Grid, face_map: np.ndarray
    ) -> pp.BoundaryConditionVectorial:
        """Obtain a representation of a boundary condition for a subgrid of the original
        grid.

        This is somehow better fit for the BoundaryCondition class, but it is not clear
        whether the implementation is sufficiently general to be put there.

        Parameters:
            bc: Boundary condition for the original grid.
            sub_g: Grid for which the new condition applies. Is assumed to be a subgrid
                of the grid to initialize this object.
            face_map: Index of faces of the original grid from which the new conditions
            should be picked.

        Returns:
            New boundary conditions aimed at a smaller grid. Will have type of boundary
            condition, basis and robin_weight copied from the specified faces in the
            original grid.

        """

        sub_bc = pp.BoundaryConditionVectorial(sub_g)
        for dim in range(bc.dim):
            sub_bc.is_dir[dim] = bc.is_dir[dim, face_map]
            sub_bc.is_rob[dim] = bc.is_rob[dim, face_map]
            sub_bc.is_neu[dim, sub_bc.is_dir[dim] + sub_bc.is_rob[dim]] = False

        sub_bc.robin_weight = bc.robin_weight[:, :, face_map]
        sub_bc.basis = bc.basis[:, :, face_map]

        return sub_bc

    def _constit_for_subgrid(
        self, constit: pp.FourthOrderTensor, loc_cells: np.ndarray
    ) -> pp.FourthOrderTensor:
        """Extract a constitutive law for a subgrid of the original grid.

        Parameters:
            constit: Constitutive law for the original grid.
            loc_cells: Index of cells of the original grid from which the new
                constitutive law should be picked.

        Returns:
            New constitutive law aimed at a smaller grid.

        """
        # Copy stiffness tensor, and restrict to local cells
        loc_c = constit.copy()
        loc_c.values = loc_c.values[::, ::, loc_cells]
        # Also restrict the lambda and mu fields; we will copy the stiffness tensors
        # later.
        loc_c.lmbda = loc_c.lmbda[loc_cells]
        loc_c.mu = loc_c.mu[loc_cells]
        return loc_c
