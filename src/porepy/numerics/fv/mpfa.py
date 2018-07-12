"""
Implementation of the multi-point flux approximation O-method.

"""
from __future__ import division
import warnings
import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import fvutils, tpfa
from porepy.grids import partition
from porepy.params import tensor, bc, data
from porepy.utils import matrix_compression
from porepy.utils import comp_geom as cg
from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.fv import TpfaCoupling, TpfaCouplingDFN

# ------------------------------------------------------------------------------


class MpfaMixedDim(SolverMixedDim):
    def __init__(self, physics="flow"):
        self.physics = physics

        self.discr = Mpfa(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = TpfaCoupling(self.discr)

        self.solver = Coupler(self.discr, self.coupling_conditions)


# ------------------------------------------------------------------------------


class MpfaDFN(SolverMixedDim):
    def __init__(self, dim_max, physics="flow"):
        # NOTE: There is no flow along the intersections of the fractures.

        self.physics = physics
        self.dim_max = dim_max

        self.discr = Mpfa(self.physics)
        self.coupling_conditions = TpfaCouplingDFN(self.discr)

        kwargs = {"discr_ndof": self.discr.ndof, "discr_fct": self.__matrix_rhs__}
        self.solver = Coupler(coupling=self.coupling_conditions, **kwargs)
        SolverMixDim.__init__(self)

    def __matrix_rhs__(self, g, data):
        # The highest dimensional problem compute the matrix and rhs, the lower
        # dimensional problem and empty matrix. For the latter, the size of the
        # matrix is the number of cells.
        if g.dim == self.dim_max:
            return self.discr.matrix_rhs(g, data)
        else:
            ndof = self.discr.ndof(g)
            return sps.csr_matrix((ndof, ndof)), np.zeros(ndof)


# ------------------------------------------------------------------------------


class Mpfa(Solver):
    def __init__(self, physics="flow"):
        self.physics = physics

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

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data, discretize=True):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point flux
        approximation.

        The name of data in the input dictionary (data) are:
        k : second_order_tensor
            Permeability defined cell-wise.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()
        discretize (boolean, optional): Whether to discetize prior to matrix
            assembly. If False, data should already contain discretization.
            Defaults to True.

        Return
        ------
        matrix: sparse csr (g_num_cells, g_num_cells)
            Discretization matrix.
        rhs: array (g_num_cells)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        """
        if discretize:
            self.discretize(g, data)

        div = fvutils.scalar_divergence(g)
        flux = data["flux"]
        M = div * flux

        bound_flux = data["bound_flux"]

        param = data["param"]

        bc_val = param.get_bc_val(self)

        return M, self.rhs(g, bound_flux, bc_val)

    # ------------------------------------------------------------------------------#

    def rhs(self, g, bound_flux, bc_val):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using the MPFA method. See self.matrix_rhs for a detaild
        description.
        """
        div = g.cell_faces.T

        return -div * bound_flux * bc_val

    # ------------------------------------------------------------------------------#

    def discretize(self, g, data):
        """
        The name of data in the input dictionary (data) are:
        k : second_order_tensor
            Permeability defined cell-wise. If not given a identity permeability
            is assumed and a warning arised.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        """
        param = data["param"]
        k = param.get_tensor(self)
        bnd = param.get_bc(self)
        a = param.aperture

        trm, bound_flux, bp_cell, bp_face = mpfa(g, k, bnd, apertures=a)
        data["flux"] = trm
        data["bound_flux"] = bound_flux
        data["bound_pressure_cell"] = bp_cell
        data["bound_pressure_face"] = bp_face


# ------------------------------------------------------------------------------#


def mpfa(g, k, bnd, eta=None, inverter=None, apertures=None, max_memory=None, **kwargs):
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
        g (core.grids.grid): grid to be discretized
        k (core.constit.second_order_tensor) permeability tensor
        bnd (core.bc.bc) class for boundary values
        eta Location of pressure continuity point. Defaults to 1/3 for simplex
            grids, 0 otherwise. On boundary faces with Dirichlet conditions,
            eta=0 will be enforced.
        inverter (string) Block inverter to be used, either numba (default),
            cython or python. See fvutils.invert_diagonal_blocks for details.
        apertures (np.ndarray) apertures of the cells for scaling of the face
            normals.
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
        flux, bound_flux, bound_pressure_cell, bound_pressure_face = _mpfa_local(
            g, k, bnd, eta=eta, inverter=inverter, apertures=apertures
        )
    else:
        # Estimate number of partitions necessary based on prescribed memory
        # usage
        peak_mem = _estimate_peak_memory(g)
        num_part = np.ceil(peak_mem / max_memory)

        # Let partitioning module apply the best available method
        part = partition.partition(g, num_part)

        # Boundary faces on the main grid
        glob_bound_face = g.get_all_boundary_faces()

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
            loc_flux, loc_bound_flux, loc_bp_cell, loc_bp_face, loc_faces = mpfa_partial(
                g, k, bnd, eta=eta, inverter=inverter, nodes=active_nodes
            )

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

    return flux, bound_flux, bound_pressure_cell, bound_pressure_face


# ------------------------------------------------------------------------------


def mpfa_partial(
    g,
    k,
    bnd,
    eta=0,
    inverter="numba",
    cells=None,
    faces=None,
    nodes=None,
    apertures=None,
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
        g (porepy.grids.grid.Grid): grid to be discretized
        k (porepy.params.tensor.SecondOrderTensor) permeability tensor
        bnd (porepy.params.bc.BoundarCondition) class for boundary conditions
        faces (np.ndarray) faces to be considered. Intended for partial
            discretization, may change in the future
        eta Location of pressure continuity point. Should be 1/3 for simplex
            grids, 0 otherwise. On boundary faces with Dirichlet conditions,
            eta=0 will be enforced.
        inverter (string) Block inverter to be used, either numba (default),
            cython or python. See fvutils.invert_diagonal_blocks for details.
        cells (np.array, int, optional): Index of cells on which to base the
            subgrid computation. Defaults to None.
        faces (np.array, int, optional): Index of faces on which to base the
            subgrid computation. Defaults to None.
        nodes (np.array, int, optional): Index of nodes on which to base the
            subgrid computation. Defaults to None.
        apertures (np.ndarray, float, optional) apertures of the cells for scaling of the face
            normals. Defaults to None.

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
    sub_g, l2g_faces, _ = partition.extract_subgrid(g, ind)
    l2g_cells = sub_g.parent_cell_ind

    # Local parameter fields
    # Copy permeability field, and restrict to local cells
    loc_k = k.copy()
    loc_k.perm = loc_k.perm[::, ::, l2g_cells]

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
    loc_bnd = bc.BoundaryCondition(sub_g, faces=loc_bound_ind, cond=loc_cond)

    # Discretization of sub-problem
    flux_loc, bound_flux_loc, bound_pressure_cell, bound_pressure_face = _mpfa_local(
        sub_g, loc_k, loc_bnd, eta=eta, inverter=inverter, apertures=apertures
    )

    # Map to global indices
    face_map, cell_map = fvutils.map_subgrid_to_grid(
        g, l2g_faces, l2g_cells, is_vector=False
    )
    flux_glob = face_map * flux_loc * cell_map
    bound_flux_glob = face_map * bound_flux_loc * face_map.transpose()
    bound_pressure_cell_glob = face_map * bound_pressure_cell * cell_map
    bound_pressure_face_glob = face_map * bound_pressure_face * face_map.T

    # By design of mpfa, and the subgrids, the discretization will update faces
    # outside the active faces. Kill these.
    outside = np.setdiff1d(np.arange(g.num_faces), active_faces, assume_unique=True)
    flux_glob[outside, :] = 0
    bound_flux_glob[outside, :] = 0
    bound_pressure_cell_glob[outside, :] = 0
    bound_pressure_face_glob[outside, :] = 0

    return (
        flux_glob,
        bound_flux_glob,
        bound_pressure_cell_glob,
        bound_pressure_face_glob,
        active_faces,
    )


def _mpfa_local(g, k, bnd, eta=None, inverter="numba", apertures=None):
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
        (i)   A * grad_p            = 0
        (ii)  B * grad_p + C * p_cc = 0
        (iii) 0            D * p_cc = I
    Here, the first equation represents flux continuity, and involves only the
    pressure gradients (grad_p). The second equation gives pressure continuity
    over cell faces, thus B will contain distances between cell centers and the
    face continuity points, while C consists of +- 1 (depending on which side
    the cell is relative to the face normal vector). The third equation
    enforces the pressure to be unity in one cell at a time. Thus (i)-(iii) can
    be inverted to express the pressure gradients as in terms of the cell
    center variables, that is, we can compute the basis functions on the
    sub-cells. Because of the method construction (again see reference paper),
    the basis function of a cell c will be non-zero on all sub-cells sharing
    a vertex with c. Finally, the fluxes as functions of cell center values are
    computed by insertion into Darcy's law (which is essentially half of A from
    (i), that is, only consider contribution from one side of the face.
    Boundary values can be incorporated with appropriate modifications -
    Neumann conditions will have a non-zero right hand side for (i), while
    Dirichlet gives a right hand side for (ii).

    """
    if eta is None:
        eta = fvutils.determine_eta(g)

    # The method reduces to the more efficient TPFA in one dimension, so that
    # method may be called. In 0D, there is no internal discretization to be
    # done.
    if g.dim == 1:
        discr = tpfa.Tpfa()
        params = data.Parameters(g)
        params.set_bc("flow", bnd)
        params.set_aperture(apertures)
        params.set_tensor("flow", k)
        d = {"param": params}
        discr.discretize(g, d)
        return (
            d["flux"],
            d["bound_flux"],
            d["bound_pressure_cell"],
            d["bound_pressure_face"],
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
        cell_centers, face_normals, face_centers, R, _, nodes = cg.map_grid(g)
        g.cell_centers = cell_centers
        g.face_normals = face_normals
        g.face_centers = face_centers
        g.nodes = nodes

        # Rotate the permeability tensor and delete last dimension
        k = k.copy()
        k.perm = np.tensordot(R.T, np.tensordot(R, k.perm, (1, 0)), (0, 1))
        k.perm = np.delete(k.perm, (2), axis=0)
        k.perm = np.delete(k.perm, (2), axis=1)

    # Define subcell topology, that is, the local numbering of faces, subfaces,
    # sub-cells and nodes. This numbering is used throughout the
    # discretization.
    subcell_topology = fvutils.SubcellTopology(g)

    # Obtain normal_vector * k, pairings of cells and nodes (which together
    # uniquely define sub-cells, and thus index for gradients. See comment
    # below for the ordering of elements in the subcell gradient.
    nk_grad, cell_node_blocks, sub_cell_index = _tensor_vector_prod(
        g, k, subcell_topology, apertures
    )

    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for pressure continuity
    pr_cont_grad = fvutils.compute_dist_face_cell(g, subcell_topology, eta)

    # Darcy's law
    darcy = -nk_grad[subcell_topology.unique_subfno]

    # Pair fluxes over subfaces, that is, enforce conservation
    nk_grad = subcell_topology.pair_over_subfaces(nk_grad)

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1 (Depending on whether the cell is on the
    # positive or negative side of the face.
    # The .A suffix is necessary to get a numpy array, instead of a scipy
    # matrix.
    sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A
    pr_cont_cell = sps.coo_matrix(
        (sgn[0], (subcell_topology.subfno, subcell_topology.cno))
    ).tocsr()
    # The cell centers give zero contribution to flux continuity
    nk_cell = sps.coo_matrix(
        (np.zeros(1), (np.zeros(1), np.zeros(1))),
        shape=(subcell_topology.num_subfno, subcell_topology.num_cno),
    ).tocsr()
    del sgn

    # Mapping from sub-faces to faces
    hf2f = sps.coo_matrix(
        (
            np.ones(subcell_topology.unique_subfno.size),
            (subcell_topology.fno_unique, subcell_topology.subfno_unique),
        )
    )

    # Update signs
    sgn_unique = g.cell_faces[
        subcell_topology.fno_unique, subcell_topology.cno_unique
    ].A.ravel("F")

    # The boundary faces will have either a Dirichlet or Neumann condition, but
    # not both (Robin is not implemented).
    # Obtain mappings to exclude boundary faces.
    bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bnd, g.dim)

    # No flux conditions for Dirichlet boundary faces
    nk_grad = bound_exclusion.exclude_dirichlet(nk_grad)
    nk_cell = bound_exclusion.exclude_dirichlet(nk_cell)
    # No pressure condition for Neumann boundary faces
    pr_cont_grad_all = pr_cont_grad
    pr_cont_grad = bound_exclusion.exclude_neumann(pr_cont_grad)
    pr_cont_cell = bound_exclusion.exclude_neumann(pr_cont_cell)

    # So far, the local numbering has been based on the numbering scheme
    # implemented in SubcellTopology (which treats one cell at a time). For
    # efficient inversion (below), it is desirable to get the system over to a
    # block-diagonal structure, with one block centered around each vertex.
    # Obtain the necessary mappings.
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
        sub_cell_index, cell_node_blocks, subcell_topology.nno_unique, bound_exclusion
    )

    del cell_node_blocks, sub_cell_index

    # System of equations for the subcell gradient variables. On block diagonal
    # form.
    grad_eqs = sps.vstack([nk_grad, pr_cont_grad])

    num_nk_cell = nk_cell.shape[0]
    num_pr_cont_grad = pr_cont_grad.shape[0]
    del nk_grad

    grad = rows2blk_diag * grad_eqs * cols2blk_diag

    del grad_eqs
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
    # The columns of igrad corresponds to the ordering of the equations in
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
    flux = hf2f * darcy * igrad * (-sps.vstack([nk_cell, pr_cont_cell]))

    ####
    # Boundary conditions
    rhs_bound = _create_bound_rhs(
        bnd,
        bound_exclusion,
        subcell_topology,
        sgn_unique,
        g,
        num_nk_cell,
        num_pr_cont_grad,
    )
    # Discretization of boundary values
    bound_flux = hf2f * darcy * igrad * rhs_bound

    # Below here, fields necessary for reconstruction of boundary pressures

    # Diagonal matrix that divides by number of sub-faces per face
    half_face_per_face = sps.diags(1. / (hf2f * np.ones(hf2f.shape[1])))

    # Contribution to face pressure from sub-cell gradients, calculated as
    # gradient times distance. Then further map to faces, and divide by number
    # of contributions per face
    dp = (
        half_face_per_face
        * hf2f
        * pr_cont_grad_all
        * igrad
        * (-sps.vstack([nk_cell, pr_cont_cell]))
    )

    # Internal faces, and boundary faces with a Dirichle condition do not need
    # information on the gradient.
    # Implementation note: This can be expanded to pressure recovery also
    # on internal faces by including them here, and below.
    remove_not_neumann = sps.diags(bnd.is_neu.astype(np.int))
    dp = remove_not_neumann * dp

    # We also need pressure in the cell next to the boundary face.
    bound_faces = g.get_all_boundary_faces()
    # A trick to get the boundary face: We know that one element is -1 (e.g.
    # outside the domain). Add 1, sum cell indices (will only contain the
    # internal cell; the one outside is now zero), and then subtract 1 again.
    bound_cells = np.sum(g.cell_face_as_dense()[:, bound_faces] + 1, axis=0) - 1
    cell_contrib = sps.coo_matrix(
        (np.ones_like(bound_faces), (bound_faces, bound_cells)),
        shape=(g.num_faces, g.num_cells),
    )
    cell_contrib = remove_not_neumann * cell_contrib

    bound_pressure_cell = dp + cell_contrib

    sgn_arr = np.zeros(g.num_faces)
    sgn_arr[bound_faces] = g.cell_faces[bound_faces].sum(axis=1).A.ravel()
    sgn_mat = sps.diags(sgn_arr)

    bound_pressure_face_neu = (
        sgn_mat * half_face_per_face * hf2f * pr_cont_grad_all * igrad * rhs_bound
    )
    # For Dirichlet faces, simply recover the boundary condition
    bound_pressure_face_dir = sps.diags(bnd.is_dir.astype(np.int))

    bound_pressure_face = (
        bound_pressure_face_dir + remove_not_neumann * bound_pressure_face_neu
    )

    return flux, bound_flux, bound_pressure_cell, bound_pressure_face


# ----------------------------------------------------------------------------#
#
# The functions below are helper functions, which are not really necessary to
# understand in detail to use the method. They also tend to be less well
# documented.
#
# ----------------------------------------------------------------------------#


def _estimate_peak_memory(g):
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


def _tensor_vector_prod(g, k, subcell_topology, apertures=None):
    """
    Compute product of normal vectors and tensors on a sub-cell level.

    This is essentially defining Darcy's law for each sub-face in terms of
    sub-cell gradients. Thus, we also implicitly define the global ordering
    of sub-cell gradient variables (via the interpretation of the columns in
    nk).

    NOTE: In the local numbering below, in particular in the variables i and j,
    it is tacitly assumed that g.dim == g.nodes.shape[0] ==
    g.face_normals.shape[0] etc. See implementation note in main method.

    Parameters:
        g (core.grids.grid): Discretization grid
        k (core.constit.second_order_tensor): The permeability tensor
        subcell_topology (fvutils.SubcellTopology): Wrapper class containing
            subcell numbering.

    Returns:
        nk: sub-face wise product of normal vector and permeability tensor.
        cell_node_blocks pairings of node and cell indices, which together
            define a sub-cell.
        sub_cell_ind: index of all subcells

    """

    # Stack cell and nodes, and remove duplicate rows. Since subcell_mapping
    # defines cno and nno (and others) working cell-wise, this will
    # correspond to a unique rows (Matlab-style) from what I understand.
    # This also means that the pairs in cell_node_blocks uniquely defines
    # subcells, and can be used to index gradients etc.
    cell_node_blocks, blocksz = matrix_compression.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )

    nd = g.dim

    # Duplicates in [cno, nno] corresponds to different faces meeting at the
    # same node. There should be exactly nd of these. This test will fail
    # for pyramids in 3D
    assert np.all(blocksz == nd)

    # Define row and column indices to be used for normal_vectors * perm.
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a gradient) and
    # is adjusted according to block sizes
    _, j = np.meshgrid(subcell_topology.subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    j += matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)

    # Distribute faces equally on the sub-faces
    num_nodes = np.diff(g.face_nodes.indptr)
    normals = g.face_normals[:, subcell_topology.fno] / num_nodes[subcell_topology.fno]
    if apertures is not None:
        normals = normals * apertures[subcell_topology.cno]

    # Represent normals and permeability on matrix form
    ind_ptr = np.hstack((np.arange(0, j.size, nd), j.size))
    normals_mat = sps.csr_matrix((normals.ravel("F"), j.ravel("F"), ind_ptr))
    k_mat = sps.csr_matrix(
        (k.perm[::, ::, cell_node_blocks[0]].ravel("F"), j.ravel("F"), ind_ptr)
    )

    nk = normals_mat * k_mat

    # Unique sub-cell indexes are pulled from column indices, we only need
    # every nd column (since nd faces of the cell meet at each vertex)
    sub_cell_ind = j[::, 0::nd]
    return nk, cell_node_blocks, sub_cell_ind


def _block_diagonal_structure(sub_cell_index, cell_node_blocks, nno, bound_exclusion):
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
    # the equations are either of flux or pressure continuity (not both)
    nno_flux = bound_exclusion.exclude_dirichlet(nno)
    nno_pressure = bound_exclusion.exclude_neumann(nno)
    node_occ = np.hstack((nno_flux, nno_pressure))
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
        (np.ones(sub_cell_index.size), (subcind_nodes, np.arange(sub_cell_index.size)))
    ).tocsr()
    return rows2blk_diag, cols2blk_diag, size_of_blocks


def _create_bound_rhs(bnd, bound_exclusion, subcell_topology, sgn, g, num_flux, num_pr):
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

    fno = subcell_topology.fno_unique
    num_neu = np.sum(is_neu[fno])
    num_dir = np.sum(is_dir[fno])
    num_bound = num_neu + num_dir

    # Neumann boundary conditions
    # Find Neumann faces, exclude Dirichlet faces (since these are excluded
    # from the right hand side linear system), and do necessary formating.
    neu_ind = np.argwhere(
        bound_exclusion.exclude_dirichlet(is_neu[fno].astype("int64"))
    ).ravel("F")
    # We also need to map the respective Neumann and Dirichlet half-faces to
    # the global half-face numbering (also interior faces). The latter should
    # not have Dirichlet and Neumann excluded (respectively), and thus we need
    # new fields
    neu_ind_all = np.argwhere(is_neu[fno].astype("int")).ravel("F")
    dir_ind_all = np.argwhere(is_dir[fno].astype("int")).ravel("F")
    num_face_nodes = g.face_nodes.sum(axis=0).A.ravel(order="F")

    # For the Neumann boundary conditions, we define the value as seen from
    # the innside of the domain. E.g. outflow is defined to be positive. We
    # therefore set the matrix indices to -1. We also have to scale it with
    # the number of nodes per face because the flux of face is the sum of its
    # half-faces.
    scaled_sgn = -1 / num_face_nodes[fno[neu_ind_all]]
    if neu_ind.size > 0:
        neu_cell = sps.coo_matrix(
            (scaled_sgn, (neu_ind, np.arange(neu_ind.size))),
            shape=(num_flux, num_bound),
        )
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        neu_cell = sps.coo_matrix((num_flux, num_bound))

    # Dirichlet boundary conditions
    dir_ind = np.argwhere(
        bound_exclusion.exclude_neumann(is_dir[fno].astype("int64"))
    ).ravel("F")
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix(
            (sgn[dir_ind_all], (dir_ind, num_neu + np.arange(dir_ind.size))),
            shape=(num_pr, num_bound),
        )
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        dir_cell = sps.coo_matrix((num_pr, num_bound))

    # Number of elements in neu_ind and neu_ind_all are equal, we can test with
    # any of them. Same with dir.
    if neu_ind.size > 0 and dir_ind.size > 0:
        neu_dir_ind = np.hstack([neu_ind_all, dir_ind_all]).ravel("F")
    elif neu_ind.size > 0:
        neu_dir_ind = neu_ind_all
    elif dir_ind.size > 0:
        neu_dir_ind = dir_ind_all
    else:
        raise ValueError("Boundary values should be either Dirichlet or " "Neumann")

    num_subfno = subcell_topology.num_subfno_unique

    # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
    # Map these to all half-face indices
    bnd_2_all_hf = sps.coo_matrix(
        (np.ones(num_bound), (np.arange(num_bound), neu_dir_ind)),
        shape=(num_bound, num_subfno),
    )
    # The user of the discretization should now nothing about half faces,
    # thus map from half face to face indices.
    hf_2_f = sps.coo_matrix(
        (
            np.ones(subcell_topology.subfno_unique.size),
            (subcell_topology.subfno_unique, subcell_topology.fno_unique),
        ),
        shape=(num_subfno, g.num_faces),
    )
    rhs_bound = sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf * hf_2_f
    return rhs_bound
