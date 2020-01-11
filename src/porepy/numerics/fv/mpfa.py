"""
Implementation of the multi-point flux approximation O-method.

"""
from __future__ import division
import warnings
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.fv import fvutils
from porepy.numerics.fv.fv_elliptic import FVElliptic


class Mpfa(FVElliptic):
    def __init__(self, keyword):
        super(Mpfa, self).__init__(keyword)

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (pressure dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    def discretize(self, g, data):
        """
        Discretize the second order elliptic equation using multi-point flux
        approximation.

        The method computes fluxes over faces in terms of pressures defined at the
        cell centers.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]
            deviation_from_plane_tol: The geometrical tolerance, used in the check to
                rotate 2d grids

        parameter_dictionary contains the entries:
            second_order_tensor: (SecondOrderTensor) Permeability defined
                cell-wise. This is the effective permeability; any scaling of
                the permeability (such as with fracture apertures) should be
                included in the permeability.
            bc: (BoundaryCondition) boundary conditions
            vector_source: (boolean) Optional. Discretization of the divergence of
                a vector source. For example, gravity in the flow equations. If False,
                standard MPFA is used.
            mpfa_eta: (float/np.ndarray) Optional. Range [0, 1). Location of
                pressure continuity point. If not given, porepy tries to set an optimal
                value.
            reconstruction_eta: (float/np.ndarray) Optional. Range [0, 1]. Location of
                pressure reconstruction point at faces. If not given, mpfa_eta is used.
            mpfa_inverter (str): Optional. Inverter to apply for local problems.
                Can take values 'numba' (default), 'cython' or 'python'.

        matrix_dictionary will be updated with the following entries:
            flux: sps.csc_matrix (g.num_faces, g.num_cells)
                flux discretization, cell center contribution
            bound_flux: sps.csc_matrix (g.num_faces, g.num_faces)
                flux discretization, face contribution
            bound_pressure_cell: sps.csc_matrix (g.num_faces, g.num_cells)
                Operator for reconstructing the pressure trace. Cell center contribution
            bound_pressure_face: sps.csc_matrix (g.num_faces, g.num_faces)
                Operator for reconstructing the pressure trace. Face contribution
            div_vector_source: sps.csc_matrix (g.num_faces, g.num_cells*dim)
                Discretization of the flux due to vector source term, cell center contribution
                Only if vector_source is True

        Parameters
        ----------
        g (pp.Grid): grid, or a subclass, with geometry fields computed.
        data (dict): For entries, see above.
        faces (np.ndarray): optional. Defines active faces.
        """
        deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        # Extract parameters
        k = parameter_dictionary["second_order_tensor"]
        bnd = parameter_dictionary["bc"]

        vector_source = parameter_dictionary.get("mpfa_vector_source", False)

        eta = parameter_dictionary.get("mpfa_eta", None)
        eta_reconstruction = parameter_dictionary.get("reconstruction_eta", None)
        inverter = parameter_dictionary.get("mpfa_inverter", None)

        if not vector_source:
            trm, bound_flux, bp_cell, bp_face = self.mpfa(
                g,
                k,
                bnd,
                deviation_from_plane_tol,
                vector_source=vector_source,
                eta=eta,
                eta_reconstruction=eta_reconstruction,
                inverter=inverter,
            )
        else:
            trm, bound_flux, bp_cell, bp_face, div_vec_source = self.mpfa(
                g,
                k,
                bnd,
                deviation_from_plane_tol,
                vector_source=vector_source,
                eta=eta,
                eta_reconstruction=eta_reconstruction,
                inverter=inverter,
            )

            matrix_dictionary[self.div_vector_source_key] = div_vec_source

        matrix_dictionary[self.flux_matrix_key] = trm
        matrix_dictionary[self.bound_flux_matrix_key] = bound_flux
        matrix_dictionary[self.bound_pressure_cell_matrix_key] = bp_cell
        matrix_dictionary[self.bound_pressure_face_matrix_key] = bp_face

    def mpfa(
        self,
        g,
        k,
        bnd,
        deviation_from_plane_tol=1e-5,
        vector_source=False,
        eta=None,
        eta_reconstruction=None,
        inverter=None,
        max_memory=None,
        **kwargs
    ):
        """
        Discretize the scalar elliptic equation by the multi-point flux
        approximation method.
        The method computes fluxes over faces in terms of pressures in adjacent
        cells (defined as all cells sharing at least one vertex with the face).
        This corresponds to the MPFA-O method, see
        Aavatsmark (2002): An introduction to the MPFA-O method on
                quadrilateral grids, Comp. Geosci. for details.
        Implementation needs:
            1) The local linear systems should be scaled with the permeability and
            the local grid size, so that we avoid rounding errors accumulating
            under grid refinement / convergence tests.
            2) It probably makes sense to create a wrapper class to store the
            discretization, interface to linear solvers etc.
        Right now, there are concrete plans for 2).

        Parameters:
            g (pp.Grid): grid to be discretized
            k (pp.Second_order_tensor) permeability tensor. This is
                the effective permeability; for lower-dimensional grids, any
                aperture scaling of the permeability should be included in k.
            bnd (pp.BoundaryCondition) class for boundary values
            deviation_from_plane_tol: The geometrical tolerance, used in the check to
                rotate 2d grids
            vector_source: discretization of the divergence of vector source term,
                e.g. gravity in the flow equations. Default False. If True,
                vector sources are discretized as part of the discrete flux operator
            eta Location of pressure continuity point. Defaults to 1/3 for simplex
                grids, 0 otherwise. On boundary faces with Dirichlet conditions,
                eta=0 will be enforced.
            eta_reconstruction Location of pressure reconstruction point on faces.
            inverter (string) Block inverter to be used, either numba (default),
                cython or python. See fvutils.invert_diagonal_blocks for details.
            max_memory (double): Threshold for peak memory during discretization.
                If the **estimated** memory need is larger than the provided
                threshold, the discretization will be split into an appropriate
                number of sub-calculations, using mpfa_partial().

        Returns:
            scipy.sparse.csr_matrix (shape num_faces, num_cells): flux
                discretization, in the form of mapping from cell pressures to face
                fluxes.
            scipy.sparse.csr_matrix (shape num_faces, num_faces): discretization of
                boundary conditions. Interpreted as fluxes induced by the boundary
                condition (both Dirichlet and Neumann). For Neumann, this will be
                the prescribed flux over the boundary face, and possibly fluxes
                over faces having nodes on the boundary. For Dirichlet, the values
                will be fluxes induced by the prescribed pressure. Incorporation as
                a right hand side in linear system by multiplication with
                divergence operator.
            scipy.sparse.csr_matrix (shape num_faces, num_cells): Used to recover
                pressure on boundary faces. Contribution from computed cell
                pressures only; contribution from faces (below) also needed.
            scipy.sparse.csr_matrix (shape num_faces, num_faces): Used to recover
                pressure on boundary faces. Contribution from boundary conditions.
            scipy.sparse.csr_matrix (shape num_faces, num_cells*dim): discretization of
                the flux due to vector source term, in the form of mapping from cell vector
                sources to face fluxes. Only if vector_source is True
        Example:
            # Set up a Cartesian grid
            g = structured.CartGrid([5, 5])
            k = tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))
            g.compute_geometry()
            # Dirirchlet boundary conditions
            bound_faces = g.tags['domain_boundary_faces'].ravel()
            bnd = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)
            # Discretization
            flux, bound_flux, bp_cell, bp_face = mpfa(g, k, bnd)
            # Source in the middle of the domain
            q = np.zeros(g.num_cells)
            q[12] = 1
            # Divergence operator for the grid
            div = fvutils.scalar_divergence(g)
            # Discretization matrix
            A = div * flux
            # Assign boundary values to all faces on the bounary
            bound_vals = np.zeros(g.num_faces)
            bound_vals[bound_faces] = np.arange(bound_faces.size)
            # Assemble the right hand side and solve
            rhs = q + div * bound_flux * bound_vals
            x = sps.linalg.spsolve(A, rhs)
            # Recover flux
            f = flux * x - bound_flux * bound_vals
            # Recover boundary pressure
            bp = bp_cell * x + bp_face * bound_vals
        """

        if max_memory is None:
            # For the moment nothing to do here, just call main mpfa method for the
            # entire grid.
            # TODO: We may want to estimate the memory need, and give a warning if
            # this seems excessive

            if not vector_source:
                flux, bound_flux, bound_pressure_cell, bound_pressure_face = self._local_discr(
                    g,
                    k,
                    bnd,
                    deviation_from_plane_tol,
                    vector_source=vector_source,
                    eta=eta,
                    eta_reconstruction=eta_reconstruction,
                    inverter=inverter,
                )
            else:
                flux, bound_flux, bound_pressure_cell, bound_pressure_face, div_vector_source = self._local_discr(
                    g,
                    k,
                    bnd,
                    deviation_from_plane_tol,
                    vector_source=vector_source,
                    eta=eta,
                    eta_reconstruction=eta_reconstruction,
                    inverter=inverter,
                )
        else:
            # Estimate number of partitions necessary based on prescribed memory
            # usage
            peak_mem = self._estimate_peak_memory(g)
            num_part = np.ceil(peak_mem / max_memory)

            # Let partitioning module apply the best available method
            part = pp.partition.partition(g, num_part)

            # Empty fields for flux and bound_flux. Will be expanded as we go.
            # Implementation note: It should be relatively straightforward to
            # estimate the memory need of flux (face_nodes -> node_cells ->
            # unique).
            flux = sps.csr_matrix(g.num_faces, g.num_cells)
            bound_flux = sps.csr_matrix(g.num_faces, g.num_faces)
            bound_pressure_cell = sps.csr_matrix(g.num_faces, g.num_cells)
            bound_pressure_face = sps.csr_matrix(g.num_faces, g.num_faces)

            cn = g.cell_nodes()

            face_covered = np.zeros(g.num_faces, dtype=np.bool)

            for p in range(part.max()):
                # Cells in this partitioning
                cell_ind = np.argwhere(part == p).ravel("F")
                # To discretize with as little overlap as possible, we use the
                # keyword nodes to specify the update stencil. Find nodes of the
                # local cells.
                active_cells = np.zeros(g.num_cells, dtype=np.bool)
                active_cells[cell_ind] = 1
                active_nodes = np.squeeze(np.where((cn * active_cells) > 0))

                # Perform local discretization.
                if not vector_source:
                    loc_flux, loc_bound_flux, loc_bp_cell, loc_bp_face, loc_faces = self.partial_discr(
                        g,
                        k,
                        bnd,
                        deviation_from_plane_tol,
                        vector_source=vector_source,
                        eta=eta,
                        eta_reconstruction=eta_reconstruction,
                        inverter=inverter,
                        nodes=active_nodes,
                    )
                else:
                    div_vector_source = sps.csr_matrix(g.num_faces, g.num_cells * g.dim)
                    loc_flux, loc_bound_flux, loc_bp_cell, loc_bp_face, loc_div_vector_source, loc_faces = self.partial_discr(
                        g,
                        k,
                        bnd,
                        deviation_from_plane_tol,
                        vector_source=vector_source,
                        eta=eta,
                        eta_reconstruction=eta_reconstruction,
                        inverter=inverter,
                        nodes=active_nodes,
                    )
                    loc_div_vector_source[face_covered, :] *= 0
                    div_vector_source += loc_div_vector_source

                # Eliminate contribution from faces already covered
                loc_flux[face_covered, :] *= 0
                loc_bound_flux[face_covered, :] *= 0
                loc_bp_cell[face_covered, :] *= 0
                loc_bp_face[face_covered, :] *= 0

                face_covered[loc_faces] = 1

                flux += loc_flux
                bound_flux += loc_bound_flux
                bound_pressure_cell += loc_bp_cell
                bound_pressure_face += loc_bp_face

        if not vector_source:
            return flux, bound_flux, bound_pressure_cell, bound_pressure_face
        else:
            return (
                flux,
                bound_flux,
                bound_pressure_cell,
                bound_pressure_face,
                div_vector_source,
            )

    def partial_discr(
        self,
        g,
        k,
        bnd,
        deviation_from_plane_tol=1e-5,
        vector_source=False,
        eta=0,
        eta_reconstruction=None,
        inverter="numba",
        cells=None,
        faces=None,
        nodes=None,
    ):
        """
        Run an MPFA discretization on subgrid, and return discretization in terms
        of global variable numbers.

        Scenarios where the method will be used include updates of permeability,
        and the introduction of an internal boundary (e.g. fracture growth).

        The subgrid can be specified in terms of cells, faces and nodes to be
        updated. For details on the implementation, see
        fv_utils.cell_ind_for_partial_update()

        Parameters:
            g (pp.Grid): grid to be discretized
            k (pp.Second_order_tensor) permeability tensor. This is
                the effective permeability; for lower-dimensional grids, any
                aperture scaling of the permeability should be included in k.
            bnd (pp.BoundaryCondition) class for boundary values
            faces (np.ndarray) faces to be considered. Intended for partial
                discretization, may change in the future
            bnd (core.bc.bc) class for boundary values
            deviation_from_plane_tol: The geometrical tolerance, used in the check to
                rotate 2d grids
            vector_source: discretization of the divergence of a vector source,
                e.g. gravity in the flow equations. Default False. If True,
                vector sources are discretized as part of the discrete flux operator
            eta Location of pressure continuity point. Should be 1/3 for simplex
                grids, 0 otherwise. On boundary faces with Dirichlet conditions,
                eta=0 will be enforced.
            eta_reconstruction Location of pressure reconstruction point on faces.
            inverter (string) Block inverter to be used, either numba (default),
                cython or python. See fvutils.invert_diagonal_blocks for details.
            cells (np.array, int, optional): Index of cells on which to base the
                subgrid computation. Defaults to None.
            faces (np.array, int, optional): Index of faces on which to base the
                subgrid computation. Defaults to None.
            nodes (np.array, int, optional): Index of nodes on which to base the
                subgrid computation. Defaults to None.

            Note that if all of {cells, faces, nodes} are None, empty matrices will
            be returned.

        Returns:
            sps.csr_matrix (g.num_faces x g.num_cells): Flux discretization,
                computed on a subgrid.
            sps.csr_matrix (g,num_faces x g.num_faces): Boundary flux
                discretization, computed on a subgrid
            np.array (int): Global of the faces where the flux discretization is
                computed.

        """
        if cells is not None:
            warnings.warn("Cells keyword for partial mpfa has not been tested")
        if faces is not None:
            warnings.warn("Faces keyword for partial mpfa has not been tested")

        # Find computational stencil, based on specified cells, faces and nodes.
        ind, active_faces = fvutils.cell_ind_for_partial_update(
            g, cells=cells, faces=faces, nodes=nodes
        )

        # Extract subgrid, together with mappings between local and global
        # cells
        sub_g, l2g_faces, _ = pp.grids.partition.extract_subgrid(g, ind)
        l2g_cells = sub_g.parent_cell_ind

        # Local parameter fields
        # Copy permeability field, and restrict to local cells
        loc_k = k.copy()
        loc_k.values = loc_k.values[::, ::, l2g_cells]

        glob_bound_face = g.get_all_boundary_faces()

        # Boundary conditions are slightly more complex. Find local faces
        # that are on the global boundary.
        loc_bound_ind = np.argwhere(np.in1d(l2g_faces, glob_bound_face)).ravel("F")
        loc_cond = np.array(loc_bound_ind.size * ["neu"])
        # Then pick boundary condition on those faces.
        if loc_bound_ind.size > 0:
            # We could have avoided to explicitly define Neumann conditions,
            # since these are default.
            # For primal-like discretizations like the MPFA, internal boundaries
            # are handled by assigning Neumann conditions.
            is_dir = np.logical_and(bnd.is_dir, np.logical_not(bnd.is_internal))
            is_neu = np.logical_or(bnd.is_neu, bnd.is_internal)

            is_dir = is_dir[l2g_faces[loc_bound_ind]]
            is_neu = is_neu[l2g_faces[loc_bound_ind]]

            loc_cond[is_dir] = "dir"
        loc_bnd = pp.BoundaryCondition(sub_g, faces=loc_bound_ind, cond=loc_cond)

        # Map to global indices
        face_map, cell_map = fvutils.map_subgrid_to_grid(
            g, l2g_faces, l2g_cells, is_vector=False
        )

        # By design of mpfa, and the subgrids, the discretization will update faces
        # outside the active faces. Prepare for this.
        outside = np.setdiff1d(np.arange(g.num_faces), active_faces, assume_unique=True)

        # Discretization of sub-problem
        if not vector_source:
            flux_loc, bound_flux_loc, bound_pressure_cell, bound_pressure_face = self._local_discr(
                sub_g,
                loc_k,
                loc_bnd,
                deviation_from_plane_tol,
                vector_source=vector_source,
                eta=eta,
                eta_reconstruction=eta_reconstruction,
                inverter=inverter,
            )

        else:
            flux_loc, bound_flux_loc, bound_pressure_cell, bound_pressure_face, div_vector_source_loc = self._local_discr(
                sub_g,
                loc_k,
                loc_bnd,
                deviation_from_plane_tol,
                vector_source=vector_source,
                eta=eta,
                eta_reconstruction=eta_reconstruction,
                inverter=inverter,
            )
            # NOTE MS: I am not sure about this mapping since div_vector_source is ready to be
            # multiplied with a cell center vector. cell_map in this case should
            # have dimensions of (num_cells_loc * g.dim, g.num_cells * g.dim).
            # I think fvutils.map_subgrid_to_grid should be modified in order
            # to return 3 objects: face_map, cell_map, cell_map_nd
            # Anyhow, for most applications, discretization goes via _local_discr
            # this is just a reminder where issues may arise
            div_vector_source_glob = face_map * div_vector_source_loc * cell_map
            div_vector_source_glob[outside, :] = 0

        flux_glob = face_map * flux_loc * cell_map
        bound_flux_glob = face_map * bound_flux_loc * face_map.transpose()
        bound_pressure_cell_glob = face_map * bound_pressure_cell * cell_map
        bound_pressure_face_glob = face_map * bound_pressure_face * face_map.T

        # Kill faces outside the activa faces
        flux_glob[outside, :] = 0
        bound_flux_glob[outside, :] = 0
        bound_pressure_cell_glob[outside, :] = 0
        bound_pressure_face_glob[outside, :] = 0

        if not vector_source:
            return (
                flux_glob,
                bound_flux_glob,
                bound_pressure_cell_glob,
                bound_pressure_face_glob,
                active_faces,
            )
        else:
            return (
                flux_glob,
                bound_flux_glob,
                bound_pressure_cell_glob,
                bound_pressure_face_glob,
                div_vector_source_glob,
                active_faces,
            )

    def _local_discr(
        self,
        g,
        k,
        bnd,
        deviation_from_plane_tol=1e-5,
        vector_source=False,
        eta=None,
        eta_reconstruction=None,
        inverter="numba",
    ):
        """
        Actual implementation of the MPFA O-method. To calculate MPFA on a grid
        directly, either call this method, or, to respect the privacy of this
        method, the main mpfa method with no memory constraints.

        Method properties and implementation details.

        The pressure is discretized as a linear function on sub-cells (see
        reference paper). In this implementation, the pressure is represented by
        its cell center value and the sub-cell gradients (this is in contrast to
        most papers, which use auxiliary pressures on the faces; the current
        formulation is equivalent, but somewhat easier to implement).
        The method will give continuous fluxes over the faces, and pressure
        continuity for certain points (controlled by the parameter eta). This can
        be expressed as a linear system on the form

            (i)    A * grad_p              = 0
            (ii)   Ar * grad_p + Cr * p_cc = 0
            (iii)  B  * grad_p + C  * p_cc = 0
            (iv)   0             D  * p_cc = I

        Here, the first equation represents flux continuity, and involves only the
        pressure gradients (grad_p). The second equation gives the Robin conditions,
        relating flux to the pressure. The third equation gives pressure continuity
        over interior faces, thus B will contain distances between cell centers and the
        face continuity points, while C consists of +- 1 (depending on which side
        the cell is relative to the face normal vector). The fourth equation
        enforces the pressure to be unity in one cell at a time. Thus (i)-(iv) can
        be inverted to express the pressure gradients in terms of the cell
        center variables, that is, we can compute the basis functions on the
        sub-cells. Because of the method construction (again see reference paper),
        the basis function of a cell c will be non-zero on all sub-cells sharing
        a vertex with c. Finally, the fluxes as functions of cell center values are
        computed by insertion into Darcy's law (which is essentially half of A from
        (i), that is, only consider contribution from one side of the face.
        Boundary values can be incorporated with appropriate modifications -
        Neumann conditions will have a non-zero right hand side for (i), while
        Dirichlet gives a right hand side for (iii).
        """
        # Implementational note on boundary conditions: A note on the possibility of
        # subface boundary conditions in mpfa/mpsa can be found in the function
        # _mpsa_local() in the mpsa.py module.

        if eta is None:
            eta = fvutils.determine_eta(g)

        # The method reduces to the more efficient TPFA in one dimension, so that
        # method may be called. In 0D, there is no internal discretization to be
        # done.
        if g.dim == 1:
            discr = pp.Tpfa(self.keyword)
            params = pp.Parameters(g)
            params["bc"] = bnd
            params["second_order_tensor"] = k

            d = {
                pp.PARAMETERS: {self.keyword: params},
                pp.DISCRETIZATION_MATRICES: {self.keyword: {}},
            }
            discr.discretize(g, d, vector_source=vector_source)
            matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][self.keyword]
            if not vector_source:
                return (
                    matrix_dictionary[self.flux_matrix_key],
                    matrix_dictionary[self.bound_flux_matrix_key],
                    matrix_dictionary[self.bound_pressure_cell_matrix_key],
                    matrix_dictionary[self.bound_pressure_face_matrix_key],
                )
            else:
                return (
                    matrix_dictionary[self.flux_matrix_key],
                    matrix_dictionary[self.bound_flux_matrix_key],
                    matrix_dictionary[self.bound_pressure_cell_matrix_key],
                    matrix_dictionary[self.bound_pressure_face_matrix_key],
                    matrix_dictionary[self.div_vector_source_key],
                )
        elif g.dim == 0:
            return sps.csr_matrix([0]), 0, 0, 0

        # The grid coordinates are always three-dimensional, even if the grid is
        # really 2D. This means that there is not a 1-1 relation between the number
        # of coordinates of a point / vector and the real dimension. This again
        # violates some assumptions tacitly made in the discretization (in
        # particular that the number of faces of a cell that meets in a vertex
        # equals the grid dimension, and that this can be used to construct an
        # index of local variables in the discretization). These issues should be
        # possible to overcome, but for the moment, we simply force 2D grids to be
        # proper 2D.

        if g.dim == 2:
            # Rotate the grid into the xy plane and delete third dimension. First
            # make a copy to avoid alterations to the input grid
            g = g.copy()
            cell_centers, face_normals, face_centers, R, _, nodes = pp.map_geometry.map_grid(
                g, deviation_from_plane_tol
            )
            g.cell_centers = cell_centers
            g.face_normals = face_normals
            g.face_centers = face_centers
            g.nodes = nodes

            # Rotate the permeability tensor and delete last dimension
            k = k.copy()
            k.values = np.tensordot(R.T, np.tensordot(R, k.values, (1, 0)), (0, 1))
            k.values = np.delete(k.values, (2), axis=0)
            k.values = np.delete(k.values, (2), axis=1)

        # Define subcell topology, that is, the local numbering of faces, subfaces,
        # sub-cells and nodes. This numbering is used throughout the
        # discretization.
        subcell_topology = fvutils.SubcellTopology(g)

        # Below, the boundary conditions should be defined on the subfaces.
        if bnd.num_faces == subcell_topology.num_subfno_unique:
            # The boundary conditions is already given on the subfaces
            subface_rhs = True
        else:
            # If bnd is not already a sub-face_bound we extend it
            bnd = pp.fvutils.boundary_to_sub_boundary(bnd, subcell_topology)
            subface_rhs = False
        # IS: Correct this paragraph!
        # Obtain normal_vector * k, pairings of cells and nodes (which together
        # uniquely define sub-cells, and thus index for gradients. See comment
        # below for the ordering of elements in the subcell gradient.

        # The normal vectors used in the product are simply the face normals
        # (with areas downscaled to account for subfaces). The sign of
        # nk_grad_all coincides with the direction of the normal vector.
        nk_grad_all, cell_node_blocks, sub_cell_index = fvutils.scalar_tensor_vector_prod(
            g, k, subcell_topology
        )

        ## Contribution from subcell gradients to local system.
        # The pressure at a subface continuity point is given by the subcell
        # pressure plus a deviation computed as the distance
        # from cell center to continuity point times the (by construction constant)

        # Contribution from subcell gradient to pressure continuity condition.
        # The operation consists of two parts:
        # 1) The cell-face distances from subcells that share a subface
        #    are paired (put on the same row).
        # 2) The sign of the distance is changed for those subcells where the
        #    subface normal vector points into the cell. This happens both on
        #    interior subfaces, and on faces on the boundary.
        # NOTE: The second operation is reversed for Robin boundary conditions,
        #       see below.
        pr_cont_grad_paired = fvutils.compute_dist_face_cell(g, subcell_topology, eta)

        # Discretized Darcy's law: The flux over a subface is given by the
        # area weighted normal vector, multiplied with the subcell permeability,
        # and expressed in terms of the subcell gradient (to be computed)
        # For interior subfaces, Darcy's law is only computed for one adjacent
        # subcell.
        # Minus sign to get flow from high to low potential.
        darcy = -nk_grad_all[subcell_topology.unique_subfno]

        # Pair fluxes over subfaces, that is, enforce conservation.
        # The operation consists of two parts:
        # 1) The expression nK computed from subcells that share a subface
        #    is paired (put on the same row). That is, the number of rows in
        #    nk_grad is reduced.
        # 2) The sign of the nK products is changed for those subcells where the
        #    subface normal vector points into the cell. This happens both on
        #    interior subfaces, and on faces on the boundary.
        nk_grad_paired = subcell_topology.pair_over_subfaces(nk_grad_all)

        ### Contribution from cell center potentials to local systems.

        # Cell center pressure contribution to flux continuity on subfaces:
        # There is no contribution (fluxes are driven by gradients only).
        # Construct a zero matrix of the right size, that is, a mapping from
        # cells to subfaces.
        nk_cell = sps.coo_matrix(
            (np.zeros(1), (np.zeros(1), np.zeros(1))),
            shape=(subcell_topology.num_subfno, subcell_topology.num_cno),
        ).tocsr()

        # Cell center pressure contribution to pressure continuity on subfaces:
        # The contributions will be unity. For (sub)cells where the subface normal vector
        # points inwards, the contribution is -1, that is, we do the same
        # modification as for nk_grad_paired.

        # Recover the sign of the subface normal vectors relative to
        # subcells (positive if normal vector points out of subcell).
        # The information can be obtained from g.cell_faces, which doubles as a
        # discrete divergence operator.
        # The .A suffix is necessary to get a numpy array, instead of a scipy
        # matrix.
        sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A
        # Then insert the signs into a matrix that maps cells to subfaces
        pr_cont_cell_all = sps.coo_matrix(
            (sgn[0], (subcell_topology.subfno, subcell_topology.cno))
        ).tocsr()

        # Create a sign array that only contains one side (subcell) of each subface.
        # This will be used at various points below.
        sgn_unique = g.cell_faces[
            subcell_topology.fno_unique, subcell_topology.cno_unique
        ].A.ravel("F")

        ## Discretize the Robin condition.
        # This takes the form
        #
        #   f + bnd.robin_weight * pressure_face * subface_area = something.
        #
        # The scaling is important here: The f is a flux integrated over the
        # half face, thus the scaling with the subface area is necessary.
        #
        # The pressure at face (pressure trace) is reconstructed from the cell center
        # pressure plus a deviation given by the cell-face distance times the subcell
        # gradient. For the local system, these terms contribute to cell center
        # and gradient terms, respectively.
        #
        # NOTE: In this implementation, the computation of the subface pressure
        # replaces the already discretized pressure continuity condition (the
        # relevant rows are modified). As part of the modification, the change
        # of sign in pr_cont_grad_paired and pr_cont_cell_all is reversed.
        # That is, the cell pressure is scaled by a positive number, while
        # the gradient is multiplied with the vector from cell center to
        # face continuity point.

        # Compute the sign of the subcells relative to the normal vector of the
        # subface. The sign is scaled by the Robin weight, and the area of the
        # subface.
        # The conditions for subfaces not on a Robin boundary will be eliminated
        # below (in computation of pr_cont_grad).
        num_nodes = np.diff(g.face_nodes.indptr)
        sgn_scaled_by_subface_area = (
            bnd.robin_weight
            * sgn_unique
            * g.face_areas[subcell_topology.fno_unique]
            / num_nodes[subcell_topology.fno_unique]
        )
        # pair_over_subfaces flips the sign so we flip it back
        # Contribution from cell centers.
        pr_trace_grad_all = sps.diags(sgn_scaled_by_subface_area) * pr_cont_grad_paired
        # Contribution from gradient.
        pr_trace_cell_all = sps.coo_matrix(
            (
                bnd.robin_weight[subcell_topology.subfno]
                * g.face_areas[subcell_topology.fno]
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
            )
        )

        # The boundary faces will have either a Dirichlet or Neumann condition, or
        # Robin condition
        # Obtain mappings to exclude boundary faces.
        bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bnd, g.dim)

        # No flux conditions for Dirichlet and Robin boundary faces
        nk_grad_n = bound_exclusion.exclude_robin_dirichlet(nk_grad_paired)
        nk_cell = bound_exclusion.exclude_robin_dirichlet(nk_cell)

        # Robin condition is only applied to Robin boundary faces
        nk_grad_r = bound_exclusion.keep_robin(nk_grad_paired)
        pr_trace_grad = bound_exclusion.keep_robin(pr_trace_grad_all)
        pr_trace_cell = bound_exclusion.keep_robin(pr_trace_cell_all)

        # No pressure condition for Neumann or Robin boundary faces
        pr_cont_grad = bound_exclusion.exclude_neumann_robin(pr_cont_grad_paired)
        pr_cont_cell = bound_exclusion.exclude_neumann_robin(pr_cont_cell_all)

        # The discretization is now in a sense complete: We have discretized
        # the flux and pressure continuity equations, with the necessary
        # adjustments for boundary conditions. The resulting linear system
        # takes the form
        #
        #  (nk_grad_n                , nk_cell  )
        #  (nk_grad_r - pr_trace_grad, -pr_trace_cell) (subcell_grad)    = RHS
        #  (pr_cont_grad             , pr_cont_cell  ) (center_pressure)
        #
        # The first equation enforces flux continuity (with nk_cell=0)
        # The second equation enforces Robin boundary conditions
        # The third equation enforces pressure continuity.
        # RHS is detailed below, this enforces calculation of the discrete
        # basis function.
        #
        # The next step is to compute the subcell gradients as a function of
        # the ultimate degrees of freedom in the discretization scheme, that is
        # cell center pressures and boundary conditions. For cell center
        # pressures, assign a unit pressure in one cell center at a time
        # while assigning zero pressures in other cells (this is implemented
        # as moving the second column in the above matrix to the right hand side).
        # When solving for subcell_grad, this gives the gradients, in effect,
        # the boundary conditions.
        #
        # Implementation of boundary conditions need special mentioning:
        # First, the rows corresponding to boundary subfaces are modified:
        #
        #  i) On Neumann subfaces, only the first block row is kept, second and
        #     third row are eliminated (there is no Robin or pressure continuity
        #     condition).
        #  ii) Robin conditions have dedicated rows, these subfaces are eliminated
        #      from the first and third row.
        #  iii) For Dirichlet subfaces, only the third block row is kept.
        #
        # In each case, the boundary subface only has contribution from one
        # IS: Close the parenthesis!
        # adjacent cell. By defining an appropriate RHS column (say, unit flux
        # on a subface, with all subfaces having zero value, the subscale
        # gradient for this boundary condition is found [in practice this means
        # continuity of flux and pressure for internal subfaces, on boundary
        # subfaces it enforces zero flux (Neumann), pressure (Dirichlet) or
        # combination (Robin). That is, the boundary dofs do not interfere with
        # each other].
        #
        # As an implementation note, the fastest approach found is to explicitly
        # calculate the inverse of the matrix
        #
        #  (nk_grad_n)
        #  (nk_grad_r - pr_trace_grad)
        #  (pr_cont_grad)
        #
        # The calculation exploits the block diagonal structure of the matrix
        # (after a permutation that gathers all columns and rows associated
        # with a single vertex, that is, an interaction region). Fast inverses
        # can then be used. After inversion, subscale gradients can be found
        # for any right hand side by a matrix-vector product.

        # System of equations for the subcell gradient variables. On block diagonal
        # form.
        # NOTE: I think in the discretization for sub_cells a flow out of the cell is
        # negative. This is a contradiction to what is done for the boundary conditions
        # where we want to set dot(n, flux) where n is the normal pointing outwards.
        # thats why we need +nk_grad_r - pr_trace_grad -pr_trace_cell = rhs
        # instead of how we would expect: -nk_grad_r + pr_trace_grad +pr_trace_cell= rhs.
        # This is also why we multiply with -1 in scaled_sgn in _create_bound_rhs
        # EK: Not really, The nk_grad terms are calculated according to what
        # happens to be the normal vector of the face.
        # IS: Should we delete these two comments?
        grad_eqs = sps.vstack([nk_grad_n, nk_grad_r - pr_trace_grad, pr_cont_grad])

        num_nk_cell = nk_cell.shape[0]
        num_nk_rob = nk_grad_r.shape[0]
        num_pr_cont_grad = pr_cont_grad.shape[0]

        del nk_grad_n, nk_grad_r, pr_trace_grad

        # So far, the local numbering has been based on the numbering scheme
        # implemented in SubcellTopology (which treats one cell at a time). For
        # efficient inversion (below), it is desirable to get the system over to a
        # block-diagonal structure, with one block centered around each vertex.
        # Obtain the necessary mappings.
        rows2blk_diag, cols2blk_diag, size_of_blocks = self._block_diagonal_structure(
            sub_cell_index,
            cell_node_blocks,
            subcell_topology.nno_unique,
            bound_exclusion,
        )

        # Re-organize system into a block-diagonal form
        grad = rows2blk_diag * grad_eqs * cols2blk_diag

        del grad_eqs
        # Invert the system, and map back to the original form
        igrad = (
            cols2blk_diag
            * fvutils.invert_diagonal_blocks(grad, size_of_blocks, method=inverter)
            * rows2blk_diag
        )

        del grad, cols2blk_diag, rows2blk_diag

        # Technical note: The elements in igrad are organized as follows:
        # The fields subcell_topology.cno and .nno will together identify Nd
        # placements in igrad that are associated with the same cell and the same
        # node, that is, they belong to the same subcell. These placements are used
        # to store the discrete gradient of that cell, with the first item
        # representing the x-component etc.
        # As an example, to find the gradient in the subcell of cell ci, associated
        # with node ni, first find the indexes of subcell_topology.cno and .nno
        # that contain ci and ni, respectively. The first of these indexes give the
        # row of the x-component of the gradient, the second the y-component etc.
        #
        # The columns of igrad correspond to the ordering of the equations in
        # grad; as recovered in _block_diagonal_structure. In practice, the first
        # columns correspond to unit pressures assigned to faces (as used for
        # boundary conditions or to discretize discontinuities over internal faces,
        # say, to represent heterogeneous gravity), while the latter group
        # gives gradients induced by cell center pressures.
        #
        # Note tacit assumptions: 1) Each cell has exactly Nd faces meeting in a
        # vertex; or else, there would not be an exact match between the
        # number of equal (nno-cno) pairs and the number of components in the
        # gradient. This assumption is always okay in 2d, in 3d it rules out cells
        # shaped as pyramids, in which case mpfa is not defined without making
        # further specifications of the method.
        # 2) The number of components in the gradient is equal to the spatial
        # dimension of the grid, as defined in g.dim. Thus 2d grids embedded in 3d
        # will run into trouble, unless the grid is first projected down to its
        # natural plane. This can be fixed by a more general implementation, but
        # it would require quite deep changes to the code.

        # Flux discretization:
        # Negative in front of sps.vstack comes from moving the cell center
        # unknown in the discretized sytsem to the right hand side.
        # The negative in front of pr_trace_cell comes from the grad_eqs
        rhs_cells = -sps.vstack([nk_cell, -pr_trace_cell, pr_cont_cell])
        flux = darcy * igrad * rhs_cells

        # Boundary conditions
        rhs_bound = self._create_bound_rhs(
            bnd,
            bound_exclusion,
            subcell_topology,
            sgn_unique,
            g,
            num_nk_cell,
            num_nk_rob,
            num_pr_cont_grad,
            subface_rhs,
        )
        # Discretization of boundary values
        bound_flux = darcy * igrad * rhs_bound

        # Obtain the reconstruction of the pressure
        if eta_reconstruction is None:
            # If no reconstruction eta is given, use the continuity points
            eta_reconstruction = eta

        dist_grad, cell_centers = reconstruct_presssure(
            g, subcell_topology, eta_reconstruction
        )

        pressure_trace_cell = dist_grad * igrad * rhs_cells + cell_centers
        pressure_trace_bound = dist_grad * igrad * rhs_bound

        if not subface_rhs:
            # In this case we set the value at a face, thus, we need to distribute the
            # face values to the subfaces. We do this by an area-weighted average. The flux
            # will in this case be integrated over the faces, that is:
            # flux *\cdot * normal * face_area
            area_scaling = 1.0 / (hf2f * np.ones(hf2f.shape[1]))
            area_mat = sps.diags(hf2f.T * area_scaling)

            bound_flux = hf2f * bound_flux * hf2f.T
            flux = hf2f * flux
            pressure_trace_bound = hf2f * area_mat * pressure_trace_bound * hf2f.T
            pressure_trace_cell = hf2f * area_mat * pressure_trace_cell

        if not vector_source:
            return flux, bound_flux, pressure_trace_cell, pressure_trace_bound
        else:
            # in this case, we consider the full darcy's law in the presence of
            # a vector source, e.g. gravity.
            # discr_div_vector_source is the discretised vector source, which is interpreted
            # as a force on a subface due to imbalance in cell-center vector sources.
            # This term is computed on a sub-cell basis
            # and has dimensions (num_subfaces, num_subcells * nd)
            discr_div_vector_source = self.discretize_div_vector_source(
                g,
                subcell_topology,
                bound_exclusion,
                darcy,
                igrad,
                nk_grad_all,
                nk_grad_paired,
            )

            # Output should be on cell-level (not sub-cell)
            sc2c = fvutils.cell_vector_to_subcell(
                g.dim, sub_cell_index, cell_node_blocks[0]
            )

            div_vector_source = hf2f * discr_div_vector_source * sc2c
            return (
                flux,
                bound_flux,
                pressure_trace_cell,
                pressure_trace_bound,
                div_vector_source,
            )

    def discretize_div_vector_source(
        self,
        g,
        subcell_topology,
        bound_exclusion,
        darcy,
        igrad,
        nk_grad_all,
        nk_grad_paired,
    ):
        """
        Consistent discretization of the divergence of the vector source term
        in MPFA-O method. An example of a vector source is the gravitational
        forces in Darcy's law.
        For more details, see Starnoni et al (2019), Consistent discretization
        of flow for inhomogeneoug gravitational fields, WRR

        Parameters:
            g (core.grids.grid): grid to be discretized
            subcell_topology: Wrapper class for numbering of subcell faces, cells
                etc.
            bound_exclusion: Object that can eliminate faces related to boundary
                conditions.
            darcy: discretized Darcy's law
            igrad: inverse gradient
            nk_grad_all: nK products on a sub-cell level
            nk_grad_paired: nK products after pairing

        Returns:
            scipy.sparse.csr_matrix (shape num_subfaces, num_subcells * nd):
            discretization of the vector source term, interpreted as a force on a subface
            due to imbalance in cell-center vector sources

        Method properties and implementation details.
        Basis functions, namely 'flux' and 'bound_flux', for the pressure
        discretization are obtained as in standard MPFA-O method.
        Vector source is represented as forces in the cells.
        However, jumps in vector sources over a cell face act as flux
        imbalance, and thus induce additional pressure gradients in the sub-cells.
        An additional system is set up, which applies non-zero conditions to the
        flux continuity equation. This can be expressed as a linear system on the form
            (i)   A * grad_p            = I
            (ii)  B * grad_p + C * p_cc = 0
            (iii) 0            D * p_cc = 0
        Thus (i)-(iii) can be inverted to express the additional pressure gradients
        due to imbalance in vector sources as in terms of the cell center variables.
        Thus we can compute the basis functions 'div_vector_source_jumps' on the sub-cells.
        To ensure flux continuity, as soon as a convention is chosen for what side
        the flux evaluation should be considered on, an additional term, called
        'div_vector_source_faces', is added to the full flux. This latter term represents the flux
        due to cell-center vector source acting on the face from the chosen side.
        The pair subfno_unique-unique_subfno gives the side convention.
        The full flux on the face is therefore given by
        q = flux * p + bound_flux * p_b + (div_vector_source_jumps + div_vector_source_faces) * vector_source

        Output: div_vector_source = div_vector_source_jumps + div_vector_source_faces

        The strategy is as follows.
        1. assemble r.h.s. for the new linear system, needed for the term 'div_vector_source_jumps'
        2. compute term 'div_vector_source_faces'
        """

        num_subfno = subcell_topology.num_subfno
        num_subfno_unique = subcell_topology.subfno_unique.size

        # Step 1
        # The vector source term in the flux continuity equation is discretized
        # as a force on the faces. The right hand side is thus formed of the
        # unit vector.
        vals = np.ones(num_subfno_unique)
        rows = subcell_topology.subfno_unique
        cols = subcell_topology.subfno_unique
        rhs_units = sps.coo_matrix(
            (vals, (rows, cols)), shape=(num_subfno_unique, num_subfno_unique)
        )

        # The vector source term is added to all internal faces, all Neumann faces
        # and all Robin faces.
        rhs_units_n = bound_exclusion.exclude_robin_dirichlet(rhs_units)
        # Robin condition is only applied to Robin boundary faces
        rhs_units_r = bound_exclusion.keep_robin(rhs_units)
        # The Robin condition is added after all flux equations (internal and Neumann
        # faces)
        rhs_units = sps.vstack([rhs_units_n, rhs_units_r])

        # No right hand side for cell pressure equations.
        num_dir_subface = (
            bound_exclusion.exclude_neu_rob.shape[1]
            - bound_exclusion.exclude_neu_rob.shape[0]
        )

        rhs_units_pres_var = sps.coo_matrix(
            (num_subfno - num_dir_subface, num_subfno_unique)
        )

        rhs_units = -sps.vstack([rhs_units, rhs_units_pres_var])

        del rhs_units_pres_var

        # prepare for computation of imbalance coefficients,
        # that is jumps in cell-centers vector sources, ready to be
        # multiplied with inverse gradients
        div_vector_source_jumps = -darcy * igrad * rhs_units * nk_grad_paired

        # Step 2

        # mapping from subface to unique subface for scalar problems.
        # This mapping gives the convention from which side
        # the force should be evaluated on.
        map_unique_subfno = sps.coo_matrix(
            (
                np.ones(num_subfno_unique),
                (subcell_topology.subfno_unique, subcell_topology.unique_subfno),
            )
        )

        # Prepare for computation of div_vector_source_faces term
        div_vector_source_faces = map_unique_subfno * nk_grad_all

        return div_vector_source_jumps + div_vector_source_faces

    """
    The functions below are helper functions, which are not really necessary to
    understand in detail to use the method. They also tend to be less well
    documented.
    """

    def _estimate_peak_memory(self, g):
        """
        Rough estimate of peak memory need
        """
        nd = g.dim
        num_cell_nodes = g.cell_nodes().toarray().sum(axis=1)

        # Number of unknowns around a vertex: nd per cell that share the vertex for
        # pressure gradients, and one per cell (cell center pressure)
        num_grad_unknowns = nd * num_cell_nodes

        # The most expensive field is the storage of igrad, which is block diagonal
        # with num_grad_unknowns sized blocks
        igrad_size = num_grad_unknowns.sum()

        # The discretization of Darcy's law will require nd (that is, a gradient)
        # per sub-face.
        num_sub_face = g.face_nodes.toarray().sum()
        darcy_size = nd * num_sub_face

        # Balancing of fluxes will require 2*nd (gradient on both sides) fields per
        # sub-face
        nk_grad_size = 2 * nd * num_sub_face
        # Similarly, pressure continuity requires 2 * (nd+1) (gradient on both
        # sides, and cell center pressures) numbers
        pr_cont_size = 2 * (nd + 1) * num_sub_face

        total_size = igrad_size + darcy_size + nk_grad_size + pr_cont_size

        # Not covered yet is various fields on subcell topology, mapping matrices
        # between local and block ordering etc.
        return total_size

    def _block_diagonal_structure(
        self, sub_cell_index, cell_node_blocks, nno, bound_exclusion
    ):
        """ Define matrices to turn linear system into block-diagonal form
        Parameters
        ----------
        sub_cell_index
        cell_node_blocks: pairs of cell and node pairs, which defines sub-cells
        nno node numbers associated with balance equations
        exclude_dirichlet mapping to remove rows associated with flux boundary
        exclude_neumann mapping to remove rows associated with pressure boundary
        Returns
        -------
        rows2blk_diag transform rows of linear system to block-diagonal form
        cols2blk_diag transform columns of linear system to block-diagonal form
        size_of_blocks number of equations in each block
        """

        # Stack node numbers of equations on top of each other, and sort them to
        # get block-structure. First eliminate node numbers at the boundary, where
        # the equations are either of flux, pressure continuity or robin
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

        # cell_node_blocks[1] contains the node numbers associated with each
        # sub-cell gradient (and so column of the local linear systems). A sort
        # of these will give a block-diagonal structure
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
        bnd,
        bound_exclusion,
        subcell_topology,
        sgn,
        g,
        num_flux,
        num_rob,
        num_pr,
        subface_rhs,
    ):
        """
        Define rhs matrix to get basis functions for incorporates boundary
        conditions
        Parameters
        ----------
        bnd
        exclude_dirichlet
        exclude_neumann
        fno
        sgn : +-1, defining here and there of the faces
        g : grid
        num_flux : number of equations for flux continuity
        num_pr: number of equations for pressure continuity
        Returns
        -------
        rhs_bound: Matrix that can be multiplied with inverse block matrix to get
                   basis functions for boundary values
        """
        # For primal-like discretizations like the MPFA, internal boundaries
        # are handled by assigning Neumann conditions.
        is_dir = np.logical_and(bnd.is_dir, np.logical_not(bnd.is_internal))
        is_neu = np.logical_or(bnd.is_neu, bnd.is_internal)
        is_rob = np.logical_and(bnd.is_rob, np.logical_not(bnd.is_internal))

        fno = subcell_topology.fno_unique
        num_neu = np.sum(is_neu)
        num_dir = np.sum(is_dir)
        if not num_rob == np.sum(is_rob):
            raise AssertionError()

        num_bound = num_neu + num_dir + num_rob

        # Neumann and Robin boundary conditions. Neumann and Robin conditions
        # are essentially the same for the rhs (the rhs for both is a flux).
        # However, we need to be carefull and get the indexing correct as seen
        # from the local system, that is, first Neumann, then Robin and last
        # Dirichlet.
        # Find Neumann and Robin faces, exclude Dirichlet faces (since these are excluded
        # from the right hand side linear system), and do necessary formating.
        neu_ind = np.argwhere(
            bound_exclusion.exclude_robin_dirichlet(is_neu.astype("int64"))
        ).ravel("F")
        rob_ind = np.argwhere(bound_exclusion.keep_robin(is_rob.astype("int64"))).ravel(
            "F"
        )
        neu_rob_ind = np.argwhere(
            bound_exclusion.exclude_dirichlet((is_rob + is_neu).astype("int64"))
        ).ravel("F")

        # We also need to map the respective Neumann, Robin, and Dirichlet half-faces to
        # the global half-face numbering (also interior faces). The latter should
        # not have Dirichlet and Neumann excluded (respectively), and thus we need
        # new fields
        neu_ind_all = np.argwhere(is_neu.astype("int")).ravel("F")
        rob_ind_all = np.argwhere(is_rob.astype("int")).ravel("F")
        dir_ind_all = np.argwhere(is_dir.astype("int")).ravel("F")
        num_face_nodes = g.face_nodes.sum(axis=0).A.ravel(order="F")

        # We now merge the neuman and robin indices since they are treated equivalent
        if rob_ind.size == 0:
            neu_rob_ind = neu_ind
        elif neu_ind.size == 0:
            neu_rob_ind = rob_ind + num_flux
        else:
            neu_rob_ind = np.hstack((neu_ind, rob_ind + num_flux))
        neu_rob_ind_all = np.hstack((neu_ind_all, rob_ind_all))

        # For the Neumann/Robin boundary conditions, we define the value as seen from
        # the innside of the domain. E.g. outflow is defined to be positive. We
        # therefore set the matrix indices to -1. We also have to scale it with
        # the number of nodes per face because the flux of face is the sum of its
        # half-faces.
        #
        # EK: My understanding of the multiple -1s in the flux equation for boundary conditions:
        # 1) -nk_grad in the local system gets its sign changed if the boundary
        #    normal vector points into the cell.
        # 2) During global assembly, the flux matrix is hit by the divergence
        #    operator, which will give another -1 for cells with inwards pointing
        #    normal vector.
        # NOW WHAT?
        if subface_rhs:
            # In this case we set the rhs for the sub-faces. Note that the rhs values
            # should be integrated over the subfaces, that is
            # flux_neumann *\cdot * normal * subface_area
            scaled_sgn = -1 * np.ones(neu_rob_ind_all.size)
        else:
            # In this case we set the value at a face, thus, we need to distribute the
            #  face values to the subfaces. We do this by an area-weighted average. Note
            # that the rhs values should in this case be integrated over the faces, that is:
            # flux_neumann *\cdot * normal * face_area
            scaled_sgn = -1 / num_face_nodes[fno[neu_rob_ind_all]]

        if neu_rob_ind.size > 0:
            neu_rob_cell = sps.coo_matrix(
                (scaled_sgn, (neu_rob_ind, np.arange(neu_rob_ind.size))),
                shape=(num_flux + num_rob, num_bound),
            )
        else:
            # Special handling when no elements are found. Not sure if this is
            # necessary, or if it is me being stupid
            neu_rob_cell = sps.coo_matrix((num_flux + num_rob, num_bound))

        # Dirichlet boundary conditions
        dir_ind = np.argwhere(
            bound_exclusion.exclude_neumann_robin(is_dir.astype("int64"))
        ).ravel("F")
        if dir_ind.size > 0:
            dir_cell = sps.coo_matrix(
                (
                    sgn[dir_ind_all],
                    (dir_ind, num_neu + num_rob + np.arange(dir_ind.size)),
                ),
                shape=(num_pr, num_bound),
            )
        else:
            # Special handling when no elements are found. Not sure if this is
            # necessary, or if it is me being stupid
            dir_cell = sps.coo_matrix((num_pr, num_bound))

        # Number of elements in neu_ind and neu_ind_all are equal, we can test with
        # any of them. Same with dir.
        if neu_rob_ind.size > 0 and dir_ind.size > 0:
            neu_rob_dir_ind = np.hstack([neu_rob_ind_all, dir_ind_all]).ravel("F")
        elif neu_rob_ind.size > 0:
            neu_rob_dir_ind = neu_rob_ind_all
        elif dir_ind.size > 0:
            neu_rob_dir_ind = dir_ind_all
        else:
            raise ValueError("Boundary values should be either Dirichlet or " "Neumann")

        num_subfno = subcell_topology.num_subfno_unique

        # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
        # Map these to all half-face indices
        bnd_2_all_hf = sps.coo_matrix(
            (np.ones(num_bound), (np.arange(num_bound), neu_rob_dir_ind)),
            shape=(num_bound, num_subfno),
        )

        rhs_bound = sps.vstack([neu_rob_cell, dir_cell]) * bnd_2_all_hf

        return rhs_bound


def reconstruct_presssure(g, subcell_topology, eta):
    """
    Function for reconstructing the pressure at the half faces given the
    local gradients. For a subcell Ks associated with cell K and node s, the
    pressure at a point x is given by
    p_Ks + G_Ks (x - x_k),
    x_K is the cell center of cell k. The point at which we evaluate the pressure
    is given by eta, which is equivalent to the continuity points in mpfa.
    For an internal subface we will obtain two values for the pressure,
    one for each of the cells associated with the subface. The pressure given
    here is the average of the two. Note that at the continuity points the two
    pressures will by construction be equal.

    Parameters:
        g: Grid
        subcell_topology: Wrapper class for numbering of subcell faces, cells
            etc.
        eta (float or ndarray, range=[0,1)): Optional. Parameter determining the point
            at which the pressures is evaluated. If eta is a nd-array it should be on
            the size of subcell_topology.num_subfno. If eta is not given the method will
            call fvutils.determine_eta(g) to set it.
    Returns:
        scipy.sparse.csr_matrix (num_sub_faces, num_cells):
            pressure reconstruction for the displacement at the half faces. This is
            the contribution from the cell-center pressures.
        scipy.sparse.csr_matrix (num_sub_faces, num_faces):
            Pressure reconstruction for the pressures at the half faces.
            This is the contribution from the boundary conditions.
    """

    if eta is None:
        eta = pp.fvutils.determine_eta(g)

    # Calculate the distance from the cell centers to continuity points
    D_g = pp.fvutils.compute_dist_face_cell(
        g, subcell_topology, eta, return_paired=False
    )
    # We here average the contribution on internal sub-faces.
    # If you want to get out both displacements on a sub-face your can remove
    # the averaging.
    _, IC, counts = np.unique(
        subcell_topology.subfno, return_inverse=True, return_counts=True
    )

    avg_over_subfaces = sps.coo_matrix(
        (1 / counts[IC], (subcell_topology.subfno, subcell_topology.subhfno))
    )
    D_g = avg_over_subfaces * D_g
    D_g = D_g.tocsr()

    # Get a mapping from cell centers to half-faces
    D_c = sps.coo_matrix(
        (1 / counts[IC], (subcell_topology.subfno, subcell_topology.cno))
    ).tocsr()

    return D_g, D_c
