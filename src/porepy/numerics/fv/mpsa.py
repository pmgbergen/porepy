"""

Implementation of the multi-point stress appoximation method, and also terms
related to poro-elastic coupling.

The methods are very similar to those of the MPFA method, although vector
equations tend to become slightly more complex thus, it may be useful to confer
that module as well.

"""
import warnings
import numpy as np
import scipy.sparse as sps
import logging

from porepy.numerics.fv import fvutils
from porepy.utils import matrix_compression, mcolon, sparse_mat
from porepy.grids import structured, partition
from porepy.params import tensor, bc
from porepy.numerics.mixed_dim.solver import Solver

# Module-wide logger
logger = logging.getLogger(__name__)


class Mpsa(Solver):
    def __init__(self, physics="mechanics"):
        self.physics = physics

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells times dimension (stress dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.dim * g.num_cells

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data, discretize=True):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point stress
        approximation.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()
        discretize (boolean, optional): default True. Whether to discetize
            prior to matrix assembly. If False, data should already contain
            discretization.

        Return
        ------
        matrix: sparse csr (g.dim * g_num_cells, g.dim * g_num_cells)
            Discretization matrix.
        rhs: array (g.dim * g_num_cells)
            Right-hand side which contains the boundary conditions and the scalar
            source term.
        """
        if discretize:
            self.discretize(g, data)
        div = fvutils.vector_divergence(g)
        stress = data["stress"]
        bound_stress = data["bound_stress"]
        M = div * stress

        f = data["param"].get_source(self)
        bc_val = data["param"].get_bc_val(self)

        return M, self.rhs(g, bound_stress, bc_val, f)

    # ------------------------------------------------------------------------------#

    def discretize(self, g, data):
        """
        Discretize the vector elliptic equation by the multi-point stress

        The method computes fluxes over faces in terms of displacements in
        adjacent cells (defined as the two cells sharing the face).

        The name of data in the input dictionary (data) are:
        param : Parameter(Class). Contains the following parameters:
            tensor : fourth_order_tensor
                Permeability defined cell-wise. If not given a identity permeability
                is assumed and a warning arised.
            bc : boundary conditions (optional)
            bc_val : dictionary (optional)
                Values of the boundary conditions. The dictionary has at most the
                following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
                conditions, respectively.
            apertures : (np.ndarray) (optional) apertures of the cells for scaling of
                the face normals.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.
        """

        c = data["param"].get_tensor(self)
        bnd = data["param"].get_bc(self)

        partial = data.get("partial_update", False)
        if not partial:
            stress, bound_stress = mpsa(g, c, bnd)
            data["stress"] = stress
            data["bound_stress"] = bound_stress
        else:
            a = data["param"].aperture
            fvutils.partial_discretization(
                g, data, c, bnd, a, mpsa_partial, physics=self.physics
            )

    # ------------------------------------------------------------------------------#

    def rhs(self, g, bound_stress, bc_val, f):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using the MPSA method. See self.matrix_rhs for a detailed
        description.
        """
        div = fvutils.vector_divergence(g)

        return -div * bound_stress * bc_val - f


# ------------------------------------------------------------------------------#


class FracturedMpsa(Mpsa):
    """
    Subclass of MPSA for discretizing a fractured domain. Adds DOFs on each
    fracture face which describe the fracture deformation.
    """

    def __init__(self, given_traction=False, **kwargs):
        Mpsa.__init__(self, **kwargs)
        assert hasattr(self, "physics"), "Mpsa must assign physics"
        self.given_traction_flag = given_traction

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells times dimension (stress dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        num_fracs = np.sum(g.tags["fracture_faces"])
        return g.dim * (g.num_cells + num_fracs)

    def matrix_rhs(self, g, data, discretize=True):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point stress
        approximation with dofs added on the fracture interfaces.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()
        discretize (boolean, optional): default True. Whether to discetize
            prior to matrix assembly. If False, data should already contain
            discretization.

        Return
        ------
        matrix: sparse csr (g.dim * g_num_cells + 2 * {#of fracture faces},
                            2 * {#of fracture faces})
            Discretization matrix.
        rhs: array (g.dim * g_num_cells  + g.dim * num_frac_faces)
            Right-hand side which contains the boundary conditions and the scalar
            source term.
        """
        if discretize:
            self.discretize_fractures(g, data)

        stress = data["stress"]
        bound_stress = data["bound_stress"]
        b_e = data["b_e"]
        A_e = data["A_e"]

        if self.given_traction_flag:
            L, b_l = self.given_traction(g, stress, bound_stress)
        else:
            L, b_l = self.given_slip_distance(g, stress, bound_stress)

        bc_val = data["param"].get_bc_val(self)

        frac_faces = np.matlib.repmat(g.tags["fracture_faces"], g.dim, 1)
        if data["param"].get_bc(self).bc_type == "scalar":
            frac_faces = frac_faces.ravel("F")

        elif data["param"].get_bc(self).bc_type == "vectorial":
            bc_val = bc_val.ravel("F")
        else:
            raise ValueError("Unknown boundary type")

        slip_distance = data["param"].get_slip_distance()

        A = sps.vstack((A_e, L), format="csr")
        rhs = np.hstack((b_e * bc_val, b_l * (slip_distance + bc_val)))

        return A, rhs

    def rhs(self, g, data):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point stress
        approximation with dofs added on the fracture interfaces.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()
        discretize (boolean, optional): default True. Whether to discetize
            prior to matrix assembly. If False, data should already contain
            discretization.

        Return
        ------
        matrix: sparse csr (g.dim * g_num_cells + 2 * {#of fracture faces},
                            2 * {#of fracture faces})
            Discretization matrix.
        rhs: array (g.dim * g_num_cells  + g.dim * num_frac_faces)
            Right-hand side which contains the boundary conditions and the scalar
            source term.
        """
        stress = data["stress"]
        bound_stress = data["bound_stress"]
        b_e = data["b_e"]

        if self.given_traction_flag:
            _, b_l = self.given_traction(g, stress, bound_stress)
        else:
            _, b_l = self.given_slip_distance(g, stress, bound_stress)

        bc_val = data["param"].get_bc_val(self)

        frac_faces = np.matlib.repmat(g.tags["fracture_faces"], 3, 1)
        if data["param"].get_bc(self).bc_type == "scalar":
            frac_faces = frac_faces.ravel("F")

        elif data["param"].get_bc(self).bc_type == "vectorial":
            bc_val = bc_val.ravel("F")
        else:
            raise ValueError("Unknown boundary type")

        slip_distance = data["param"].get_slip_distance()

        rhs = np.hstack((b_e * bc_val, b_l * (slip_distance + bc_val)))

        return rhs

    def traction(self, g, data, sol):
        """
        Extract the traction on the faces from fractured fv solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        sol : array (g.dim * (g.num_cells + {#of fracture faces}))
            Solution, stored as [cell_disp, fracture_disp]

        Return
        ------
        T : array (g.dim * g.num_faces)
            traction on each face

        """
        bc_val = data["param"].get_bc_val(self.physics).copy()
        frac_disp = self.extract_frac_u(g, sol)
        cell_disp = self.extract_u(g, sol)

        frac_faces = (g.frac_pairs).ravel("C")

        if data["param"].get_bc(self).bc_type == "vectorial":
            bc_val = bc_val.ravel("F")

        frac_ind = mcolon.mcolon(g.dim * frac_faces, g.dim * frac_faces + g.dim)
        bc_val[frac_ind] = frac_disp

        T = data["stress"] * cell_disp + data["bound_stress"] * bc_val
        return T

    def extract_u(self, g, sol):
        """  Extract the cell displacement from fractured fv solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        sol : array (g.dim * (g.num_cells + {#of fracture faces}))
            Solution, stored as [cell_disp, fracture_disp]

        Return
        ------
        u : array (g.dim * g.num_cells)
            displacement at each cell

        """
        # pylint: disable=invalid-name
        return sol[: g.dim * g.num_cells]

    def extract_frac_u(self, g, sol):
        """  Extract the fracture displacement from fractured fv solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        sol : array (g.dim * (g.num_cells + {#of fracture faces}))
            Solution, stored as [cell_disp, fracture_disp]

        Return
        ------
        u : array (g.dim *{#of fracture faces})
            displacement at each fracture face

        """
        # pylint: disable=invalid-name
        return sol[g.dim * g.num_cells :]

    def discretize_fractures(self, g, data, faces=None, **kwargs):
        """
        Discretize the vector elliptic equation by the multi-point stress and added
        degrees of freedom on the fracture faces

        The method computes fluxes over faces in terms of displacements in
        adjacent cells (defined as the two cells sharing the face).

        The name of data in the input dictionary (data) are:
        param : Parameter(Class). Contains the following parameters:
            tensor : fourth_order_tensor
                Permeability defined cell-wise. If not given a identity permeability
                is assumed and a warning arised.
            bc : boundary conditions (optional)
            bc_val : dictionary (optional)
                Values of the boundary conditions. The dictionary has at most the
                following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
                conditions, respectively.
            apertures : (np.ndarray) (optional) apertures of the cells for scaling of
                the face normals.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.
        """

        #    dir_bound = g.get_all_boundary_faces()
        #    bound = bc.BoundaryCondition(g, dir_bound, ['dir'] * dir_bound.size)

        frac_faces = g.tags["fracture_faces"]

        bound = data["param"].get_bc(self)
        is_dir = bound.is_dir

        if bound.bc_type == "scalar":
            if not np.all(is_dir[frac_faces]):
                is_dir[frac_faces] = True
                bound = bc.BoundaryCondition(g, is_dir, "dir")
        elif bound.bc_type == "vectorial":
            if not np.all(is_dir[:, frac_faces]):
                is_dir[:, frac_faces] = True
                bound = bc.BoundaryConditionVectorial(g, is_dir, "dir")
        else:
            raise ValueError("Unknow boundary condition type: " + bound.bc_type)

        # Discretize with normal mpsa
        self.discretize(g, data, **kwargs)
        stress, bound_stress = data["stress"], data["bound_stress"]
        # Create A and rhs
        div = fvutils.vector_divergence(g)
        a = div * stress
        b = div * bound_stress

        # we find the matrix indices of the fracture
        if faces is None:
            frac_faces = g.frac_pairs
            frac_faces_left = frac_faces[0]
            frac_faces_right = frac_faces[1]
        else:
            raise NotImplementedError("not implemented given faces")

        int_b_left = mcolon.mcolon(
            g.dim * frac_faces_left, g.dim * frac_faces_left + g.dim
        )
        int_b_right = mcolon.mcolon(
            g.dim * frac_faces_right, g.dim * frac_faces_right + g.dim
        )
        int_b_ind = np.ravel((int_b_left, int_b_right), "C")

        # We find the sign of the left and right faces.
        sgn_left = _sign_matrix(g, frac_faces_left)
        sgn_right = _sign_matrix(g, frac_faces_right)
        # The displacement on the internal boundary face are considered unknowns,
        # so we move them over to the lhs. The rhs now only consists of the
        # external boundary faces
        b_internal = b[:, int_b_ind]
        b_external = b.copy()
        sparse_mat.zero_columns(b_external, int_b_ind)

        bound_stress_external = bound_stress.copy().tocsc()
        sparse_mat.zero_columns(bound_stress_external, int_b_ind)
        # We assume that the traction on the left hand side is equal but
        # opisite

        frac_stress_diff = (
            sgn_left * bound_stress[int_b_left, :]
            + sgn_right * bound_stress[int_b_right, :]
        )[:, int_b_ind]
        internal_stress = sps.hstack(
            (
                sgn_left * stress[int_b_left, :] + sgn_right * stress[int_b_right, :],
                frac_stress_diff,
            )
        )

        A = sps.vstack((sps.hstack((a, b_internal)), internal_stress), format="csr")
        # negative sign since we have moved b_external from lhs to rhs
        d_b = -b_external
        # sps.csr_matrix((int_b_left.size, g.num_faces * g.dim))
        d_t = (
            -sgn_left * bound_stress_external[int_b_left]
            - sgn_right * bound_stress_external[int_b_right]
        )

        b_matrix = sps.vstack((d_b, d_t), format="csr")

        data["b_e"] = b_matrix
        data["A_e"] = A

    def given_traction(self, g, stress, bound_stress, faces=None, **kwargs):
        # we find the matrix indices of the fracture
        if faces is None:
            frac_faces = g.frac_pairs
            frac_faces_left = frac_faces[0]
            frac_faces_right = frac_faces[1]
        else:
            raise NotImplementedError("not implemented given faces")

        int_b_left = mcolon.mcolon(
            g.dim * frac_faces_left, g.dim * frac_faces_left + g.dim
        )
        int_b_right = mcolon.mcolon(
            g.dim * frac_faces_right, g.dim * frac_faces_right + g.dim
        )
        int_b_ind = np.ravel((int_b_left, int_b_right), "C")

        # We find the sign of the left and right faces.
        sgn_left = _sign_matrix(g, frac_faces_left)
        sgn_right = _sign_matrix(g, frac_faces_right)

        # We obtain the stress from boundary conditions on the domain boundary
        bound_stress_external = bound_stress.copy().tocsc()
        sparse_mat.zero_columns(bound_stress_external, int_b_ind)
        bound_stress_external = bound_stress_external.tocsc()

        # We construct the L matrix, i.e., we set the traction on the left
        # fracture side
        frac_stress = (sgn_left * bound_stress[int_b_left, :])[:, int_b_ind]

        L = sps.hstack((sgn_left * stress[int_b_left, :], frac_stress))

        # negative sign since we have moved b_external from lhs to rhs
        d_t = (
            sps.csr_matrix(
                (np.ones(int_b_left.size), (np.arange(int_b_left.size), int_b_left)),
                (int_b_left.size, g.num_faces * g.dim),
            )
            - sgn_left * bound_stress_external[int_b_left]
        )  # \
        #        + sgn_right * bound_stress_external[int_b_right]

        return L, d_t

    def given_slip_distance(self, g, stress, bound_stress, faces=None):
        # we find the matrix indices of the fracture
        if faces is None:
            frac_faces = g.frac_pairs
            frac_faces_left = frac_faces[0]
            frac_faces_right = frac_faces[1]
        else:
            raise NotImplementedError("not implemented given faces")

        int_b_left = mcolon.mcolon(
            g.dim * frac_faces_left, g.dim * frac_faces_left + g.dim
        )
        int_b_right = mcolon.mcolon(
            g.dim * frac_faces_right, g.dim * frac_faces_right + g.dim
        )
        int_b_ind = np.ravel((int_b_left, int_b_right), "C")

        # We construct the L matrix, by assuming that the relative displacement
        # is given
        L = sps.hstack(
            (
                sps.csr_matrix((int_b_left.size, g.dim * g.num_cells)),
                sps.identity(int_b_left.size),
                -sps.identity(int_b_right.size),
            )
        )

        d_f = sps.csr_matrix(
            (np.ones(int_b_left.size), (np.arange(int_b_left.size), int_b_left)),
            (int_b_left.size, g.num_faces * g.dim),
        )

        return L, d_f


# ------------------------------------------------------------------------------#


def mpsa(g, constit, bound, eta=None, inverter=None, max_memory=None, **kwargs):
    """
    Discretize the vector elliptic equation by the multi-point stress
    approximation method, specifically the weakly symmetric MPSA-W method.

    The method computes stresses over faces in terms of displacments in
    adjacent cells (defined as all cells sharing at least one vertex with the
    face).  This corresponds to the MPSA-W method, see

    Keilegavlen, Nordbotten: Finite volume methods for elasticity with weak
        symmetry. Int J Num. Meth. Eng. doi: 10.1002/nme.5538.

    Implementation needs:
        1) The local linear systems should be scaled with the elastic moduli
        and the local grid size, so that we avoid rounding errors accumulating
        under grid refinement / convergence tests.
        2) It should be possible to do a partial update of the discretization
        stensil (say, if we introduce an internal boundary, or modify the
        permeability field).
        3) For large grids, the current implementation will run into memory
        issues, due to the construction of a block diagonal matrix. This can be
        overcome by splitting the discretization into several partial updates.
        4) It probably makes sense to create a wrapper class to store the
        discretization, interface to linear solvers etc.
    Right now, there are concrete plans for 2) - 4).

    Parameters:
        g (core.grids.grid): grid to be discretized
        constit (core.bc.bc) class for boundary values
        eta Location of pressure continuity point. Should be 1/3 for simplex
            grids, 0 otherwise. On boundary faces with Dirichlet conditions,
            eta=0 will be enforced.
        inverter (string) Block inverter to be used, either numba (default),
            cython or python. See fvutils.invert_diagonal_blocks for details.
        max_memory (double): Threshold for peak memory during discretization.
            If the **estimated** memory need is larger than the provided
            threshold, the discretization will be split into an appropriate
            number of sub-calculations, using mpsa_partial().

    Returns:
        scipy.sparse.csr_matrix (shape num_faces, num_cells): stress
            discretization, in the form of mapping from cell displacement to
            face stresses.
            NOTE: The cell displacements are ordered cellwise (first u_x_1,
            u_y_1, u_x_2 etc)
        scipy.sparse.csr_matrix (shape num_faces, num_faces): discretization of
            boundary conditions. Interpreted as istresses induced by the boundary
            condition (both Dirichlet and Neumann). For Neumann, this will be
            the prescribed stress over the boundary face, and possibly stress
            on faces having nodes on the boundary. For Dirichlet, the values
            will be stresses induced by the prescribed displacement.
            Incorporation as a right hand side in linear system by
            multiplication with divergence operator.
            NOTE: The stresses are ordered facewise (first s_x_1, s_y_1 etc)

    Example:
        # Set up a Cartesian grid
        g = structured.CartGrid([5, 5])
        c =tensor.FourthOrderTensor(g.dim, np.ones(g.num_cells))

        # Dirirchlet boundary conditions
        bound_faces = g.get_all_boundary_faces().ravel()
        bnd = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)

        # Discretization
        stress, bound_stress = mpsa(g, c, bnd)

        # Source in the middle of the domain
        q = np.zeros(g.num_cells * g.dim)
        q[12 * g.dim] = 1

        # Divergence operator for the grid
        div = fvutils.vector_divergence(g)

        # Discretization matrix
        A = div * stress

        # Assign boundary values to all faces on the bounary
        bound_vals = np.zeros(g.num_faces * g.dim)
        bound_vals[bound_faces] = np.arange(bound_faces.size * g.dim)

        # Assemble the right hand side and solve
        rhs = -q - div * bound_stress * bound_vals
        x = sps.linalg.spsolve(A, rhs)
        s = stress * x + bound_stress * bound_vals

    """
    if eta is None:
        eta = fvutils.determine_eta(g)

    if max_memory is None:
        # For the moment nothing to do here, just call main mpfa method for the
        # entire grid.
        # TODO: We may want to estimate the memory need, and give a warning if
        # this seems excessive
        stress, bound_stress = _mpsa_local(
            g, constit, bound, eta=eta, inverter=inverter
        )
    else:
        # Estimate number of partitions necessary based on prescribed memory
        # usage
        peak_mem = _estimate_peak_memory_mpsa(g)
        num_part = np.ceil(peak_mem / max_memory)

        logger.info("Split MPSA discretization into " + str(num_part) + " parts")

        # Let partitioning module apply the best available method
        part = partition.partition(g, num_part)

        # Empty fields for stress and bound_stress. Will be expanded as we go.
        # Implementation note: It should be relatively straightforward to
        # estimate the memory need of stress (face_nodes -> node_cells ->
        # unique).
        stress = sps.csr_matrix((g.num_faces * g.dim, g.num_cells * g.dim))
        bound_stress = sps.csr_matrix((g.num_faces * g.dim, g.num_faces * g.dim))

        cn = g.cell_nodes()

        face_covered = np.zeros(g.num_faces, dtype=np.bool)

        for p in np.unique(part):
            # Cells in this partitioning
            cell_ind = np.argwhere(part == p).ravel("F")
            # To discretize with as little overlap as possible, we use the
            # keyword nodes to specify the update stencil. Find nodes of the
            # local cells.
            active_cells = np.zeros(g.num_cells, dtype=np.bool)
            active_cells[cell_ind] = 1
            active_nodes = np.squeeze(np.where((cn * active_cells) > 0))

            # Perform local discretization.
            loc_stress, loc_bound_stress, loc_faces = mpsa_partial(
                g, constit, bound, eta=eta, inverter=inverter, nodes=active_nodes
            )

            # Eliminate contribution from faces already covered
            eliminate_ind = fvutils.expand_indices_nd(face_covered, g.dim)
            fvutils.zero_out_sparse_rows(loc_stress, eliminate_ind)
            fvutils.zero_out_sparse_rows(loc_bound_stress, eliminate_ind)

            face_covered[loc_faces] = 1

            stress += loc_stress
            bound_stress += loc_bound_stress

    return stress, bound_stress


def mpsa_partial(
    g,
    constit,
    bound,
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
        constit (porepy.params.tensor.SecondOrderTensor) permeability tensor
        bnd (porepy.params.bc.BoundaryCondition) class for boundary conditions
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
        apertures (np.array, int, optional): Cell apertures. Defaults to None.
            Unused for now, added for similarity to mpfa_partial.

        Note that if all of {cells, faces, nodes} are None, empty matrices will
        be returned.

    Returns:
        sps.csr_matrix (g.num_faces x g.num_cells): Stress discretization,
            computed on a subgrid.
        sps.csr_matrix (g,num_faces x g.num_faces): Boundary stress
            discretization, computed on a subgrid
        np.array (int): Global of the faces where the stress discretization is
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
    if (ind.size + active_faces.size) == 0:
        stress_glob = sps.csr_matrix(
            (g.dim * g.num_faces, g.dim * g.num_cells), dtype="float64"
        )
        bound_stress_glob = sps.csr_matrix(
            (g.dim * g.num_faces, g.dim * g.num_faces), dtype="float64"
        )
        return stress_glob, bound_stress_glob, active_faces
    # Extract subgrid, together with mappings between local and global
    # cells
    sub_g, l2g_faces, _ = partition.extract_subgrid(g, ind)
    l2g_cells = sub_g.parent_cell_ind

    # Copy stiffness tensor, and restrict to local cells
    loc_c = constit.copy()
    loc_c.c = loc_c.c[::, ::, l2g_cells]
    # Also restrict the lambda and mu fields; we will copy the stiffness
    # tensors later.
    loc_c.lmbda = loc_c.lmbda[l2g_cells]
    loc_c.mu = loc_c.mu[l2g_cells]

    glob_bound_face = g.get_all_boundary_faces()

    # Boundary conditions are slightly more complex. Find local faces
    # that are on the global boundary.
    loc_bound_ind = np.argwhere(np.in1d(l2g_faces, glob_bound_face)).ravel("F")

    # Then transfer boundary condition on those faces.
    loc_cond = np.array(loc_bound_ind.size * ["neu"])
    if loc_bound_ind.size > 0:
        # Neumann condition is default, so only Dirichlet needs to be set
        is_dir = bound.is_dir[l2g_faces[loc_bound_ind]]
        loc_cond[is_dir] = "dir"

    loc_bnd = bc.BoundaryCondition(sub_g, faces=loc_bound_ind, cond=loc_cond)

    # Discretization of sub-problem
    stress_loc, bound_stress_loc = _mpsa_local(
        sub_g, loc_c, loc_bnd, eta=eta, inverter=inverter
    )

    face_map, cell_map = fvutils.map_subgrid_to_grid(
        g, l2g_faces, l2g_cells, is_vector=True
    )

    # Update global face fields.
    stress_glob = face_map * stress_loc * cell_map
    bound_stress_glob = face_map * bound_stress_loc * face_map.transpose()

    # By design of mpfa, and the subgrids, the discretization will update faces
    # outside the active faces. Kill these.
    outside = np.setdiff1d(np.arange(g.num_faces), active_faces, assume_unique=True)
    eliminate_ind = fvutils.expand_indices_nd(outside, g.dim)
    fvutils.zero_out_sparse_rows(stress_glob, eliminate_ind)
    fvutils.zero_out_sparse_rows(bound_stress_glob, eliminate_ind)

    return stress_glob, bound_stress_glob, active_faces


def _mpsa_local(g, constit, bound, eta=0, inverter="numba"):
    """
    Actual implementation of the MPSA W-method. To calculate the MPSA
    discretization on a grid, either call this method, or, to respect the
    privacy of this method, call the main mpsa method with no memory
    constraints.

    Implementation details:

    The displacement is discretized as a linear function on sub-cells (see
    reference paper). In this implementation, the displacement is represented by
    its cell center value and the sub-cell gradients.

    The method will give continuous stresses over the faces, and displacement
    continuity for certain points (controlled by the parameter eta). This can
    be expressed as a linear system on the form

        (i)   A * grad_u            = 0
        (ii)  B * grad_u + C * u_cc = 0
        (iii) 0            D * u_cc = I

    Here, the first equation represents stress continuity, and involves only
    the displacement gradients (grad_u). The second equation gives displacement
    continuity over cell faces, thus B will contain distances between cell
    centers and the face continuity points, while C consists of +- 1 (depending
    on which side the cell is relative to the face normal vector). The third
    equation enforces the displacement to be unity in one cell at a time. Thus
    (i)-(iii) can be inverted to express the displacement gradients as in terms
    of the cell center variables, that is, we can compute the basis functions
    on the sub-cells. Because of the method construction (again see reference
    paper), the basis function of a cell c will be non-zero on all sub-cells
    sharing a vertex with c. Finally, the fluxes as functions of cell center
    values are computed by insertion into Hook's law (which is essentially half
    of A from (i), that is, only consider contribution from one side of the
    face.

    Boundary values can be incorporated with appropriate modifications -
    Neumann conditions will have a non-zero right hand side for (i), while
    Dirichlet gives a right hand side for (ii).

    """

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
        g = g.copy()
        g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
        g.face_centers = np.delete(g.face_centers, (2), axis=0)
        g.face_normals = np.delete(g.face_normals, (2), axis=0)
        g.nodes = np.delete(g.nodes, (2), axis=0)

        constit = constit.copy()
        constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=0)
        constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=1)

    nd = g.dim

    # Define subcell topology
    subcell_topology = fvutils.SubcellTopology(g)
    # Obtain mappings to exclude boundary faces
    bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bound, nd)
    # Most of the work is done by submethod for elasticity (which is common for
    # elasticity and poro-elasticity).

    hook, igrad, rhs_cells, _, _ = mpsa_elasticity(
        g, constit, subcell_topology, bound_exclusion, eta, inverter
    )

    hook_igrad = hook * igrad
    # NOTE: This is the point where we expect to reach peak memory need.
    del hook, igrad

    # Output should be on face-level (not sub-face)
    hf2f = fvutils.map_hf_2_f(
        subcell_topology.fno_unique, subcell_topology.subfno_unique, nd
    )

    # Stress discretization
    stress = hf2f * hook_igrad * rhs_cells

    # Right hand side for boundary discretization
    if bound_exclusion.bc_type == "scalar":
        rhs_bound = create_bound_rhs(bound, bound_exclusion, subcell_topology, g)
    elif bound_exclusion.bc_type == "vectorial":
        rhs_bound = create_bound_rhs_nd(bound, bound_exclusion, subcell_topology, g)

    # Discretization of boundary values
    bound_stress = hf2f * hook_igrad * rhs_bound
    stress, bound_stress = _zero_neu_rows(g, stress, bound_stress, bound)

    return stress, bound_stress


def mpsa_elasticity(g, constit, subcell_topology, bound_exclusion, eta, inverter):
    """
    This is the function where the real discretization takes place. It contains
    the parts that are common for elasticity and poro-elasticity, and was thus
    separated out as a helper function.

    The steps in the discretization are the same as in mpfa (although with
    everything being somewhat more complex since this is a vector equation).
    The mpfa function is currently more clean, so confer that for additional
    comments.

    Parameters:
        g: Grid
        constit: Constitutive law
        subcell_topology: Wrapper class for numbering of subcell faces, cells
            etc.
        bound_exclusion: Object that can eliminate faces related to boundary
            conditions.
        eta: Parameter determining the continuity point
        inverter: Parameter determining which method to use for inverting the
            local systems

    Returns:
        hook: Hooks law, ready to be multiplied with inverse gradients
        igrad: Inverse gradients
        rhs_cells: Right hand side used to get basis functions in terms of cell
            center displacements
        cell_node_blocks: Relation between cells and vertexes, used to group
            equations in linear system.
        hook_normal: Hooks law for the term div(I*p) in poro-elasticity
    """

    nd = g.dim

    # Compute product between normal vectors and stiffness matrices
    ncsym, ncasym, cell_node_blocks, sub_cell_index = _tensor_vector_prod(
        g, constit, subcell_topology
    )

    # Prepare for computation of forces due to cell center pressures (the term
    # div(I*p) in poro-elasticity equations. hook_normal will be used as a right
    # hand side by the biot disretization, but needs to be computed here, since
    # this is where we have access to the relevant data.
    ind_f = np.argsort(np.tile(subcell_topology.subhfno, nd), kind="mergesort")
    hook_normal = sps.coo_matrix(
        (np.ones(ind_f.size), (np.arange(ind_f.size), ind_f)),
        shape=(ind_f.size, ind_f.size),
    ) * (ncsym + ncasym)

    del ind_f
    # The final expression of Hook's law will involve deformation gradients
    # on one side of the faces only; eliminate the other one.
    # Note that this must be done before we can pair forces from the two
    # sides of the faces.
    hook = __unique_hooks_law(ncsym, ncasym, subcell_topology, nd)

    del ncasym

    # Pair the forces from each side
    ncsym = subcell_topology.pair_over_subfaces_nd(ncsym)
    ncsym = bound_exclusion.exclude_dirichlet_nd(ncsym)

    num_subfno = subcell_topology.subfno.max() + 1
    hook_cell = sps.coo_matrix(
        (np.zeros(1), (np.zeros(1), np.zeros(1))),
        shape=(num_subfno * nd, (np.max(subcell_topology.cno) + 1) * nd),
    ).tocsr()

    hook_cell = bound_exclusion.exclude_dirichlet_nd(hook_cell)

    # Book keeping
    num_sub_cells = cell_node_blocks[0].size

    d_cont_grad, d_cont_cell = __get_displacement_submatrices(
        g, subcell_topology, eta, num_sub_cells, bound_exclusion
    )

    grad_eqs = sps.vstack([ncsym, d_cont_grad])
    del ncsym, d_cont_grad

    igrad = _inverse_gradient(
        grad_eqs,
        sub_cell_index,
        cell_node_blocks,
        subcell_topology.nno_unique,
        bound_exclusion,
        nd,
        inverter,
    )

    # Right hand side for cell center variables
    rhs_cells = -sps.vstack([hook_cell, d_cont_cell])
    return hook, igrad, rhs_cells, cell_node_blocks, hook_normal


# -----------------------------------------------------------------------------
#
# Below here are helper functions, which tend to be less than well documented.
#
# -----------------------------------------------------------------------------


def _estimate_peak_memory_mpsa(g):
    """ Rough estimate of peak memory need for mpsa discretization.
    """
    nd = g.dim
    num_cell_nodes = g.cell_nodes().sum(axis=1).A

    # Number of unknowns around a vertex: nd^2 per cell that share the vertex
    # for pressure gradients, and one per cell (cell center pressure)
    num_grad_unknowns = nd ** 2 * num_cell_nodes

    # The most expensive field is the storage of igrad, which is block diagonal
    # with num_grad_unknowns sized blocks. The number of elements is the square
    # of the local system size. The factor 2 accounts for matrix storage in
    # sparse format (rows and data; ignore columns since this is in compressed
    # format)
    igrad_size = np.power(num_grad_unknowns, 2).sum() * 2

    # The discretization of Hook's law will require nd^2 (that is, a gradient)
    # per sub-face per dimension
    num_sub_face = g.face_nodes.sum()
    hook_size = nd * num_sub_face * nd ** 2

    # Balancing of stresses will require 2*nd**2 (gradient on both sides)
    # fields per sub-face per dimension
    nk_grad_size = 2 * nd * num_sub_face * nd ** 2
    # Similarly, pressure continuity requires 2 * (nd+1) (gradient on both
    # sides, and cell center pressures) numbers
    pr_cont_size = 2 * (nd ** 2 + 1) * num_sub_face * nd

    total_size = igrad_size + hook_size + nk_grad_size + pr_cont_size

    # Not covered yet is various fields on subcell topology, mapping matrices
    # between local and block ordering etc.
    return total_size


def __get_displacement_submatrices(
    g, subcell_topology, eta, num_sub_cells, bound_exclusion
):
    nd = g.dim
    # Distance from cell centers to face centers, this will be the
    # contribution from gradient unknown to equations for displacement
    # continuity
    d_cont_grad = fvutils.compute_dist_face_cell(g, subcell_topology, eta)

    # For force balance, displacements and stresses on the two sides of the
    # matrices must be paired
    d_cont_grad = sps.kron(sps.eye(nd), d_cont_grad)

    # Contribution from cell center potentials to local systems
    d_cont_cell = __cell_variable_contribution(g, subcell_topology)

    # Expand equations for displacement balance, and eliminate rows
    # associated with neumann boundary conditions
    d_cont_grad = bound_exclusion.exclude_neumann_nd(d_cont_grad)
    d_cont_cell = bound_exclusion.exclude_neumann_nd(d_cont_cell)

    # The column ordering of the displacement equilibrium equations are
    # formed as a Kronecker product of scalar equations. Bring them to the
    # same form as that applied in the force balance equations
    d_cont_grad, d_cont_cell = __rearange_columns_displacement_eqs(
        d_cont_grad, d_cont_cell, num_sub_cells, nd
    )

    return d_cont_grad, d_cont_cell


def _split_stiffness_matrix(constit):
    """
    Split the stiffness matrix into symmetric and asymetric part

    Parameters
    ----------
    constit stiffness tensor

    Returns
    -------
    csym part of stiffness tensor that enters the local calculation
    casym part of stiffness matrix not included in local calculation
    """
    dim = np.sqrt(constit.c.shape[0])

    # We do not know how constit is used outside the discretization,
    # so create deep copies to avoid overwriting. Not really sure if this is
    # necessary
    csym = 0 * constit.copy().c
    casym = constit.copy().c

    # The copy constructor for the stiffness matrix will represent all
    # dimensions as 3d. If dim==2, delete the redundant rows and columns
    if dim == 2 and csym.shape[0] == 9:
        csym = np.delete(csym, (2, 5, 6, 7, 8), axis=0)
        csym = np.delete(csym, (2, 5, 6, 7, 8), axis=1)
        casym = np.delete(casym, (2, 5, 6, 7, 8), axis=0)
        casym = np.delete(casym, (2, 5, 6, 7, 8), axis=1)

    # The splitting is hard coded based on the ordering of elements in the
    # stiffness matrix
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


def _tensor_vector_prod(g, constit, subcell_topology):
    """ Compute product between stiffness tensor and face normals.

    The method splits the stiffness matrix into a symmetric and asymmetric
    part, and computes the products with normal vectors for each. The method
    also provides a unique identification of sub-cells (in the form of pairs of
    cells and nodes), and a global numbering of subcell gradients.

    Parameters:
        g: grid
        constit: Stiffness matrix, in the form of a fourth order tensor.
        subcell_topology: Numberings of subcell quantities etc.

    Returns:
        ncsym, ncasym: Product with face normals for symmetric and asymmetric
            part of stiffness tensors. On the subcell level. In effect, these
            will be stresses on subfaces, as functions of the subcell gradients
            (to be computed somewhere else). The rows first represent stresses
            in the x-direction for all faces, then y direction etc.
        cell_nodes_blocks: Unique pairing of cell and node numbers for
            subcells. First row: Cell numbers, second node numbers. np.ndarray.
        grad_ind: Numbering scheme for subcell gradients - gives a global
            numbering for the gradients. One column per subcell, the rows gives
            the index for the individual components of the gradients.

    """

    # Stack cells and nodes, and remove duplicate rows. Since subcell_mapping
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

    # Define row and column indices to be used for normal vector matrix
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a vector) and
    # is adjusted according to block sizes
    _, cn = np.meshgrid(subcell_topology.subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    cn += matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)
    ind_ptr_n = np.hstack((np.arange(0, cn.size, nd), cn.size))

    # Distribute faces equally on the sub-faces, and store in a matrix
    num_nodes = np.diff(g.face_nodes.indptr)
    normals = g.face_normals[:, subcell_topology.fno] / num_nodes[subcell_topology.fno]
    normals_mat = sps.csr_matrix((normals.ravel("F"), cn.ravel("F"), ind_ptr_n))

    # Then row and columns for stiffness matrix. There are nd^2 elements in
    # the gradient operator, and so the structure is somewhat different from
    # the normal vectors
    _, cc = np.meshgrid(subcell_topology.subhfno, np.arange(nd ** 2))
    sum_blocksz = np.cumsum(blocksz ** 2)
    cc += matrix_compression.rldecode(sum_blocksz - blocksz[0] ** 2, blocksz)
    ind_ptr_c = np.hstack((np.arange(0, cc.size, nd ** 2), cc.size))

    # Splitt stiffness matrix into symmetric and anti-symmatric part
    sym_tensor, asym_tensor = _split_stiffness_matrix(constit)

    # Getting the right elements out of the constitutive laws was a bit
    # tricky, but the following code turned out to do the trick
    sym_tensor_swp = np.swapaxes(sym_tensor, 2, 0)
    asym_tensor_swp = np.swapaxes(asym_tensor, 2, 0)

    # The first dimension in csym and casym represent the contribution from
    # all dimensions to the stress in one dimension (in 2D, csym[0:2,:,
    # :] together gives stress in the x-direction etc.
    # Define index vector to access the right rows
    rind = np.arange(nd)

    # Empty matrices to initialize matrix-tensor products. Will be expanded
    # as we move on
    zr = np.zeros(0)
    ncsym = sps.coo_matrix((zr, (zr, zr)), shape=(0, cc.max() + 1)).tocsr()
    ncasym = sps.coo_matrix((zr, (zr, zr)), shape=(0, cc.max() + 1)).tocsr()

    # For the asymmetric part of the tensor, we will apply volume averaging.
    # Associate a volume with each sub-cell, and a node-volume as the sum of
    # all surrounding sub-cells
    num_cell_nodes = g.num_cell_nodes()
    cell_vol = g.cell_volumes / num_cell_nodes
    node_vol = (
        np.bincount(subcell_topology.nno, weights=cell_vol[subcell_topology.cno])
        / g.dim
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
        # Pick out part of Hook's law associated with this dimension
        # The code here looks nasty, it should be possible to get the right
        # format of the submatrices in a simpler way, but I couldn't do it.
        sym_dim = np.hstack(sym_tensor_swp[:, :, rind]).transpose()
        asym_dim = np.hstack(asym_tensor_swp[:, :, rind]).transpose()

        # Distribute (relevant parts of) Hook's law on subcells
        # This will be nd rows, thus cell ci is associated with indices
        # ci*nd+np.arange(nd)
        sub_cell_ind = fvutils.expand_indices_nd(cell_node_blocks[0], nd)
        sym_vals = sym_dim[sub_cell_ind]
        asym_vals = asym_dim[sub_cell_ind]

        # Represent this part of the stiffness matrix in matrix form
        csym_mat = sps.csr_matrix((sym_vals.ravel("C"), cc.ravel("F"), ind_ptr_c))
        casym_mat = sps.csr_matrix((asym_vals.ravel("C"), cc.ravel("F"), ind_ptr_c))

        # Compute average around vertexes
        casym_mat = average * casym_mat

        # Compute products of normal vectors and stiffness tensors,
        # and stack dimensions vertically
        ncsym = sps.vstack((ncsym, normals_mat * csym_mat))
        ncasym = sps.vstack((ncasym, normals_mat * casym_mat))

        # Increase index vector, so that we get rows contributing to forces
        # in the next dimension
        rind += nd

    grad_ind = cc[:, ::nd]

    return ncsym, ncasym, cell_node_blocks, grad_ind


def _inverse_gradient(
    grad_eqs,
    sub_cell_index,
    cell_node_blocks,
    nno_unique,
    bound_exclusion,
    nd,
    inverter,
):

    # Mappings to convert linear system to block diagonal form
    rows2blk_diag, cols2blk_diag, size_of_blocks = _block_diagonal_structure(
        sub_cell_index, cell_node_blocks, nno_unique, bound_exclusion, nd
    )

    grad = rows2blk_diag * grad_eqs * cols2blk_diag

    # Compute inverse gradient operator, and map back again
    igrad = (
        cols2blk_diag
        * fvutils.invert_diagonal_blocks(grad, size_of_blocks, method=inverter)
        * rows2blk_diag
    )
    return igrad


def _block_diagonal_structure(
    sub_cell_index, cell_node_blocks, nno, bound_exclusion, nd
):
    """
    Define matrices to turn linear system into block-diagonal form.

    Parameters
    ----------
    sub_cell_index
    cell_node_blocks: pairs of cell and node pairs, which defines sub-cells
    nno node numbers associated with balance equations
    exclude_dirichlet mapping to remove rows associated with stress boundary
    exclude_neumann mapping to remove rows associated with displacement boundary

    Returns
    -------
    rows2blk_diag transform rows of linear system to block-diagonal form
    cols2blk_diag transform columns of linear system to block-diagonal form
    size_of_blocks number of equations in each block
    """

    # Stack node numbers of equations on top of each other, and sort them to
    # get block-structure. First eliminate node numbers at the boundary, where
    # the equations are either of flux or pressure continuity (not both)

    nno_stress = bound_exclusion.exclude_dirichlet(nno)
    nno_displacement = bound_exclusion.exclude_neumann(nno)

    if bound_exclusion.bc_type == "scalar":
        node_occ = np.hstack((np.tile(nno_stress, nd), np.tile(nno_displacement, nd)))

    elif bound_exclusion.bc_type == "vectorial":
        node_occ = np.hstack((nno_stress, nno_displacement))

    sorted_ind = np.argsort(node_occ, kind="mergesort")
    rows2blk_diag = sps.coo_matrix(
        (np.ones(sorted_ind.size), (np.arange(sorted_ind.size), sorted_ind))
    ).tocsr()
    # Size of block systems
    sorted_nodes_rows = node_occ[sorted_ind]
    size_of_blocks = np.bincount(sorted_nodes_rows.astype("int64"))

    # cell_node_blocks[1] contains the node numbers associated with each
    # sub-cell gradient (and so column of the local linear systems). A sort
    # of these will give a block-diagonal structure
    sorted_nodes_cols = np.argsort(cell_node_blocks[1], kind="mergesort")
    subcind_nodes = sub_cell_index[::, sorted_nodes_cols].ravel("F")
    cols2blk_diag = sps.coo_matrix(
        (np.ones(sub_cell_index.size), (subcind_nodes, np.arange(sub_cell_index.size)))
    ).tocsr()
    return rows2blk_diag, cols2blk_diag, size_of_blocks


def create_bound_rhs(bound, bound_exclusion, subcell_topology, g):
    """
    Define rhs matrix to get basis functions for boundary
    conditions assigned face-wise

    Parameters
    ----------
    bound
    bound_exclusion
    fno
    sgn : +-1, defining here and there of the faces
    g : grid
    num_stress : number of equations for flux continuity
    num_displ: number of equations for pressure continuity

    Returns
    -------
    rhs_bound: Matrix that can be multiplied with inverse block matrix to get
               basis functions for boundary values
    """
    nd = g.dim
    num_stress = bound_exclusion.exclude_dir.shape[0] * nd
    num_displ = bound_exclusion.exclude_neu.shape[0] * nd
    fno = subcell_topology.fno_unique
    subfno = subcell_topology.subfno_unique
    sgn = g.cell_faces[
        subcell_topology.fno_unique, subcell_topology.cno_unique
    ].A.ravel("F")
    num_neu = sum(bound.is_neu[fno]) * nd
    num_dir = sum(bound.is_dir[fno]) * nd
    num_bound = num_neu + num_dir

    # Convenience method for duplicating a list, with a certain increment
    def expand_ind(ind, dim, increment):
        # Duplicate rows
        ind_nd = np.tile(ind, (dim, 1))
        # Add same increment to each row (0*incr, 1*incr etc.)
        ind_incr = ind_nd + increment * np.array([np.arange(dim)]).transpose()
        # Back to row vector
        ind_new = ind_incr.reshape(-1, order="F")
        return ind_new

    # Define right hand side for Neumann boundary conditions
    # First row indices in rhs matrix
    is_neu = bound_exclusion.exclude_dirichlet(bound.is_neu[fno].astype("int64"))
    neu_ind_single = np.argwhere(is_neu).ravel("F")

    # There are is_neu.size Neumann conditions per dimension
    neu_ind = expand_ind(neu_ind_single, nd, is_neu.size)

    # We also need to account for all half faces, that is, do not exclude
    # Dirichlet and Neumann boundaries.
    neu_ind_single_all = np.argwhere(bound.is_neu[fno].astype("int")).ravel("F")
    dir_ind_single_all = np.argwhere(bound.is_dir[fno].astype("int")).ravel("F")

    neu_ind_all = np.tile(neu_ind_single_all, nd)

    # Some care is needed to compute coefficients in Neumann matrix: sgn is
    # already defined according to the subcell topology [fno], while areas
    # must be drawn from the grid structure, and thus go through fno

    fno_ext = np.tile(fno, nd)
    num_face_nodes = g.face_nodes.sum(axis=0).A.ravel("F")

    # Coefficients in the matrix. For the Neumann boundary faces we set the
    # value as seen from the outside of the domain. Note that they do not
    # have to do
    # so, and we will flip the sign later. This means that a stress [1,1] on a
    # boundary face pushes(or pulls) the face to the top right corner.
    neu_val = 1 / num_face_nodes[fno_ext[neu_ind_all]]
    # The columns will be 0:neu_ind.size
    if neu_ind.size > 0:
        neu_cell = sps.coo_matrix(
            (neu_val.ravel("F"), (neu_ind, np.arange(neu_ind.size))),
            shape=(num_stress, num_bound),
        ).tocsr()

    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        neu_cell = sps.coo_matrix((num_stress, num_bound)).tocsr()

    # Dirichlet boundary conditions, procedure is similar to that for Neumann
    is_dir = bound_exclusion.exclude_neumann(bound.is_dir[fno].astype("int64"))
    dir_ind_single = np.argwhere(is_dir).ravel("F")

    dir_ind = expand_ind(dir_ind_single, nd, is_dir.size)

    # The coefficients in the matrix should be duplicated the same way as
    # the row indices, but with no increment
    dir_val = expand_ind(sgn[dir_ind_single_all], nd, 0)

    # Column numbering starts right after the last Neumann column. dir_val
    # is ordered [u_x_1, u_y_1, u_x_2, u_y_2, ...], and dir_ind shuffles this
    # ordering. The final matrix will first have the x-coponent of the displacement
    # for each face, then the y-component, etc.
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix(
            (dir_val, (dir_ind, num_neu + np.arange(dir_ind.size))),
            shape=(num_displ, num_bound),
        ).tocsr()
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        dir_cell = sps.coo_matrix((num_displ, num_bound)).tocsr()

    num_subfno = np.max(subfno) + 1

    # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
    # Map these to all half-face indices

    is_bnd = np.hstack((neu_ind_single_all, dir_ind_single_all))
    bnd_ind = fvutils.expand_indices_nd(is_bnd, nd)

    bnd_2_all_hf = sps.coo_matrix(
        (np.ones(num_bound), (np.arange(num_bound), bnd_ind)),
        shape=(num_bound, num_subfno * nd),
    )
    # The user of the discretization should now nothing about half faces,
    # thus map from half face to face indices.
    hf_2_f = fvutils.map_hf_2_f(fno, subfno, nd).transpose()
    # the rows of rhs_bound will be ordered with first the x-component of all
    # neumann faces, then the y-component of all neumann faces, then the
    # z-component of all neumann faces. Then we will have the equivalent for
    # the dirichlet faces.
    rhs_bound = sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf * hf_2_f

    return rhs_bound


def create_bound_rhs_nd(bound, bound_exclusion, subcell_topology, g):
    """
    Define rhs matrix to get basis functions for boundary
    conditions assigned component-wise.

    For parameters and return, refer to the above create_bound_rhs

    """
    nd = g.dim

    num_stress = (
        bound_exclusion.exclude_dir_x.shape[0] + bound_exclusion.exclude_dir_y.shape[0]
    )
    num_displ = (
        bound_exclusion.exclude_neu_x.shape[0] + bound_exclusion.exclude_neu_y.shape[0]
    )
    if nd == 3:
        num_stress += bound_exclusion.exclude_dir_z.shape[0]
        num_displ += bound_exclusion.exclude_neu_z.shape[0]

    fno = subcell_topology.fno_unique
    subfno = subcell_topology.subfno_unique
    sgn = g.cell_faces[
        subcell_topology.fno_unique, subcell_topology.cno_unique
    ].A.ravel("F")

    num_neu = sum(bound.is_neu[0, fno]) + sum(bound.is_neu[1, fno])
    num_dir = sum(bound.is_dir[0, fno]) + sum(bound.is_dir[1, fno])
    if nd == 3:
        num_neu += sum(bound.is_neu[2, fno])
        num_dir += sum(bound.is_dir[2, fno])

    num_bound = num_neu + num_dir

    # Define right hand side for Neumann boundary conditions
    # First row indices in rhs matrix
    is_neu_x = bound_exclusion.exclude_dirichlet_x(bound.is_neu[0, fno].astype("int64"))
    neu_ind_single_x = np.argwhere(is_neu_x).ravel("F")

    is_neu_y = bound_exclusion.exclude_dirichlet_y(bound.is_neu[1, fno].astype("int64"))
    neu_ind_single_y = np.argwhere(is_neu_y).ravel("F")
    neu_ind_single_y += is_neu_x.size

    # We also need to account for all half faces, that is, do not exclude
    # Dirichlet and Neumann boundaries.
    neu_ind_single_all_x = np.argwhere(bound.is_neu[0, fno].astype("int")).ravel("F")
    neu_ind_single_all_y = np.argwhere(bound.is_neu[1, fno].astype("int")).ravel("F")

    neu_ind_all = np.append(neu_ind_single_all_x, [neu_ind_single_all_y])

    # expand the indices
    # this procedure replaces the method 'expand_ind' in the above
    # method 'create_bound_rhs'

    # 1 - stack and sort indices

    is_bnd_neu_x = nd * neu_ind_single_all_x
    is_bnd_neu_y = nd * neu_ind_single_all_y + 1

    is_bnd_neu = np.sort(np.append(is_bnd_neu_x, [is_bnd_neu_y]))

    if nd == 3:
        is_neu_z = bound_exclusion.exclude_dirichlet_z(
            bound.is_neu[2, fno].astype("int64")
        )
        neu_ind_single_z = np.argwhere(is_neu_z).ravel("F")
        neu_ind_single_z += is_neu_x.size + is_neu_y.size

        neu_ind_single_all_z = np.argwhere(bound.is_neu[2, fno].astype("int")).ravel(
            "F"
        )

        neu_ind_all = np.append(neu_ind_all, [neu_ind_single_all_z])

        is_bnd_neu_z = nd * neu_ind_single_all_z + 2
        is_bnd_neu = np.sort(np.append(is_bnd_neu, [is_bnd_neu_z]))

    # 2 - find the indices corresponding to the boundary components
    # having Neumann condtion

    ind_is_bnd_neu_x = np.argwhere(np.isin(is_bnd_neu, is_bnd_neu_x)).ravel("F")
    ind_is_bnd_neu_y = np.argwhere(np.isin(is_bnd_neu, is_bnd_neu_y)).ravel("F")

    neu_ind_sz = ind_is_bnd_neu_x.size + ind_is_bnd_neu_y.size

    if nd == 3:
        ind_is_bnd_neu_z = np.argwhere(np.isin(is_bnd_neu, is_bnd_neu_z)).ravel("F")
        neu_ind_sz += ind_is_bnd_neu_z.size

    # 3 - create the expanded neu_ind array

    neu_ind = np.zeros(neu_ind_sz, dtype="int")

    neu_ind[ind_is_bnd_neu_x] = neu_ind_single_x
    neu_ind[ind_is_bnd_neu_y] = neu_ind_single_y
    if nd == 3:
        neu_ind[ind_is_bnd_neu_z] = neu_ind_single_z

    # Dirichlet, same procedure
    is_dir_x = bound_exclusion.exclude_neumann_x(bound.is_dir[0, fno].astype("int64"))
    dir_ind_single_x = np.argwhere(is_dir_x).ravel("F")

    is_dir_y = bound_exclusion.exclude_neumann_y(bound.is_dir[1, fno].astype("int64"))
    dir_ind_single_y = np.argwhere(is_dir_y).ravel("F")
    dir_ind_single_y += is_dir_x.size

    dir_ind_single_all_x = np.argwhere(bound.is_dir[0, fno].astype("int")).ravel("F")
    dir_ind_single_all_y = np.argwhere(bound.is_dir[1, fno].astype("int")).ravel("F")

    # expand indices

    is_bnd_dir_x = nd * dir_ind_single_all_x
    is_bnd_dir_y = nd * dir_ind_single_all_y + 1

    is_bnd_dir = np.sort(np.append(is_bnd_dir_x, [is_bnd_dir_y]))

    if nd == 3:
        is_dir_z = bound_exclusion.exclude_neumann_z(
            bound.is_dir[2, fno].astype("int64")
        )
        dir_ind_single_z = np.argwhere(is_dir_z).ravel("F")
        dir_ind_single_z += is_dir_x.size + is_dir_y.size

        dir_ind_single_all_z = np.argwhere(bound.is_dir[2, fno].astype("int")).ravel(
            "F"
        )

        is_bnd_dir_z = nd * dir_ind_single_all_z + 2
        is_bnd_dir = np.sort(np.append(is_bnd_dir, [is_bnd_dir_z]))

    ind_is_bnd_dir_x = np.argwhere(np.isin(is_bnd_dir, is_bnd_dir_x)).ravel("F")
    ind_is_bnd_dir_y = np.argwhere(np.isin(is_bnd_dir, is_bnd_dir_y)).ravel("F")

    dir_ind_sz = ind_is_bnd_dir_x.size + ind_is_bnd_dir_y.size

    if nd == 3:
        ind_is_bnd_dir_z = np.argwhere(np.isin(is_bnd_dir, is_bnd_dir_z)).ravel("F")
        dir_ind_sz += ind_is_bnd_dir_z.size

    dir_ind = np.zeros(dir_ind_sz, dtype="int")

    dir_ind[ind_is_bnd_dir_x] = dir_ind_single_x
    dir_ind[ind_is_bnd_dir_y] = dir_ind_single_y

    if nd == 3:
        dir_ind[ind_is_bnd_dir_z] = dir_ind_single_z

    # stack together
    bnd_ind = np.hstack((is_bnd_neu, is_bnd_dir))

    # Some care is needed to compute coefficients in Neumann matrix: sgn is
    # already defined according to the subcell topology [fno], while areas
    # must be drawn from the grid structure, and thus go through fno

    fno_ext = np.tile(fno, nd)
    num_face_nodes = g.face_nodes.sum(axis=0).A.ravel("F")

    # Coefficients in the matrix. For the Neumann boundary components we set the
    # value as seen from the outside of the domain. Note that they do not
    # have to do
    # so, and we will flip the sign later. This means that a stress [1,1] on a
    # boundary face pushes(or pulls) the face to the top right corner.
    neu_val = 1 / num_face_nodes[fno_ext[neu_ind_all]]

    # The columns will be 0:neu_ind.size
    if neu_ind.size > 0:
        neu_cell = sps.coo_matrix(
            (neu_val.ravel("F"), (neu_ind, np.arange(neu_ind.size))),
            shape=(num_stress, num_bound),
        ).tocsr()
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        neu_cell = sps.coo_matrix((num_stress, num_bound)).tocsr()

    # For Dirichlet, the coefficients in the matrix should be duplicated the same way as
    # the row indices, but with no increment

    dir_val_x = sgn[dir_ind_single_all_x]
    dir_val_y = sgn[dir_ind_single_all_y]

    dir_val = np.zeros(dir_ind_sz)

    dir_val[ind_is_bnd_dir_x] = dir_val_x
    dir_val[ind_is_bnd_dir_y] = dir_val_y

    if nd == 3:
        dir_val_z = sgn[dir_ind_single_all_z]
        dir_val[ind_is_bnd_dir_z] = dir_val_z

    # Column numbering starts right after the last Neumann column. dir_val
    # is ordered [u_x_1, u_y_1, u_x_2, u_y_2, ...], and dir_ind shuffles this
    # ordering. The final matrix will first have the x-coponent of the displacement
    # for each face, then the y-component, etc.
    if dir_ind.size > 0:
        dir_cell = sps.coo_matrix(
            (dir_val, (dir_ind, num_neu + np.arange(dir_ind.size))),
            shape=(num_displ, num_bound),
        ).tocsr()
    else:
        # Special handling when no elements are found. Not sure if this is
        # necessary, or if it is me being stupid
        dir_cell = sps.coo_matrix((num_displ, num_bound)).tocsr()

    num_subfno = np.max(subfno) + 1

    # The columns in neu_cell, dir_cell are ordered from 0 to num_bound-1.
    # Map these to all half-face indices

    bnd_2_all_hf = sps.coo_matrix(
        (np.ones(num_bound), (np.arange(num_bound), bnd_ind)),
        shape=(num_bound, num_subfno * nd),
    )

    # The user of the discretization should now nothing about half faces,
    # thus map from half face to face indices.

    hf_2_f = fvutils.map_hf_2_f(fno, subfno, nd).transpose()

    # the rows of rhs_bound will be ordered with first the x-component of all
    # neumann faces, then the y-component of all neumann faces, then the
    # z-component of all neumann faces. Then we will have the equivalent for
    # the dirichlet faces.

    rhs_bound = sps.vstack([neu_cell, dir_cell]) * bnd_2_all_hf * hf_2_f

    return rhs_bound


def __unique_hooks_law(csym, casym, subcell_topology, nd):
    """
    Go from products of normal vectors with stiffness matrices (symmetric
    and asymmetric), covering both sides of faces, to a discrete Hook's law,
    that, when multiplied with sub-cell gradients, will give face stresses

    Parameters
    ----------
    csym
    casym
    unique_sub_fno
    subfno
    nd

    Returns
    -------
    hook (sps.csr) nd * (nsubfno, ncells)
    """
    # unique_sub_fno covers scalar equations only. Extend indices to cover
    # multiple dimensions
    num_eqs = csym.shape[0] / nd
    ind_single = np.tile(subcell_topology.unique_subfno, (nd, 1))
    increments = np.arange(nd) * num_eqs
    ind_all = np.reshape(ind_single + increments[:, np.newaxis], -1)

    # Unique part of symmetric and asymmetric products
    hook_sym = csym[ind_all, ::]
    hook_asym = casym[ind_all, ::]

    # Hook's law, as it comes out of the normal-vector * stiffness matrix is
    # sorted with x-component balances first, then y-, etc. Sort this to a
    # face-wise ordering
    comp2face_ind = np.argsort(
        np.tile(subcell_topology.subfno_unique, nd), kind="mergesort"
    )
    comp2face = sps.coo_matrix(
        (np.ones(comp2face_ind.size), (np.arange(comp2face_ind.size), comp2face_ind)),
        shape=(comp2face_ind.size, comp2face_ind.size),
    )
    hook = comp2face * (hook_sym + hook_asym)

    return hook


def __cell_variable_contribution(g, subcell_topology):
    """
    Construct contribution from cell center variables to local systems.
    For stress equations, these are zero, while for cell centers it is +- 1
    Parameters
    ----------
    g
    fno
    cno
    subfno

    Returns
    -------

    """
    nd = g.dim
    sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A

    # Contribution from cell center potentials to local systems
    # For pressure continuity, +-1
    d_cont_cell = sps.coo_matrix(
        (sgn[0], (subcell_topology.subfno, subcell_topology.cno))
    ).tocsr()
    d_cont_cell = sps.kron(sps.eye(nd), d_cont_cell)
    # Zero contribution to stress continuity

    return d_cont_cell


def __rearange_columns_displacement_eqs(d_cont_grad, d_cont_cell, num_sub_cells, nd):
    """ Transform columns of displacement balance from increasing cell
    ordering (first x-variables of all cells, then y) to increasing
    variables (first all variables of the first cells, then...)

    Parameters
    ----------
    d_cont_grad
    d_cont_cell
    num_sub_cells
    nd
    cno

    Returns
    -------

    """
    # Repeat sub-cell indices nd times. Fortran ordering (column major)
    # gives same ordering of indices as used for the scalar equation (where
    # there are nd gradient variables for each sub-cell), and thus the
    # format of each block in d_cont_grad
    rep_ci_single_blk = np.tile(np.arange(num_sub_cells), (nd, 1)).reshape(
        -1, order="F"
    )
    # Then repeat the single-block indices nd times (corresponding to the
    # way d_cont_grad is constructed by Kronecker product), and find the
    # sorting indices
    d_cont_grad_map = np.argsort(np.tile(rep_ci_single_blk, nd), kind="mergesort")
    # Use sorting indices to bring d_cont_grad to the same order as that
    # used for the columns in the stress continuity equations
    d_cont_grad = d_cont_grad[:, d_cont_grad_map]

    # For the cell displacement variables, we only need a single expansion (
    # corresponding to the second step for the gradient unknowns)
    num_cells = d_cont_cell.shape[1] / nd
    d_cont_cell_map = np.argsort(np.tile(np.arange(num_cells), nd), kind="mergesort")
    d_cont_cell = d_cont_cell[:, d_cont_cell_map]
    return d_cont_grad, d_cont_cell


def _neu_face_sgn(g, neu_ind):
    neu_sgn = (g.cell_faces[neu_ind, :]).data
    assert (
        neu_sgn.size == neu_ind.size
    ), "A normal sign is only well defined for a boundary face"
    sort_id = np.argsort(g.cell_faces[neu_ind, :].indices)
    return neu_sgn[sort_id]


def _zero_neu_rows(g, stress, bound_stress, bnd):
    """
    We zero out all none-diagonal elements for the neumann boundary faces.
    """
    if bnd.bc_type == "scalar":
        neu_face_x = g.dim * np.ravel(np.argwhere(bnd.is_neu))
        if g.dim == 1:
            neu_face_ind = neu_face_x
        elif g.dim == 2:
            neu_face_y = neu_face_x + 1
            neu_face_ind = np.ravel((neu_face_x, neu_face_y), "F")
        elif g.dim == 3:
            neu_face_y = neu_face_x + 1
            neu_face_z = neu_face_x + 2
            neu_face_ind = np.ravel((neu_face_x, neu_face_y, neu_face_z), "F")
        else:
            raise ValueError("Only support for dimension 1, 2, or 3")
        num_neu = neu_face_ind.size

    elif bnd.bc_type == "vectorial":
        neu_face_x = g.dim * np.ravel(np.argwhere(bnd.is_neu[0, :]))
        neu_face_y = g.dim * np.ravel(np.argwhere(bnd.is_neu[1, :])) + 1
        neu_face_ind = np.sort(np.append(neu_face_x, [neu_face_y]))
        if g.dim == 2:
            pass
        elif g.dim == 3:
            neu_face_z = g.dim * np.ravel(np.argwhere(bnd.is_neu[2, :])) + 2
            neu_face_ind = np.sort(np.append(neu_face_ind, [neu_face_z]))
        else:
            raise ValueError("Only support for dimension 1, 2, or 3")
        num_neu = neu_face_ind.size

    if not num_neu:
        return stress, bound_stress

    # Frist we zero out the boundary stress. We keep the sign of the diagonal
    # element, however we discard its value (e.g. set it to +-1). The sign
    # should be negative if the nomral vector points outwards and positive if
    # the normal vector points inwards. I'm not sure if this is correct (that
    # is, zeroing out none-diagonal elements and putting the diagonal elements
    # to +-1), but it seems to give satisfactory results.
    sgn = np.sign(np.ravel(bound_stress[neu_face_ind, neu_face_ind]))
    # Set all neumann rows to zero
    bound_stress = fvutils.zero_out_sparse_rows(bound_stress, neu_face_ind, sgn)
    # For the stress matrix we zero out any rows corresponding to the Neumann
    # boundary faces (these have been moved over to the bound_stress matrix).
    stress = fvutils.zero_out_sparse_rows(stress, neu_face_ind)

    return stress, bound_stress


def _sign_matrix(g, faces):
    # We find the sign of the given faces
    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn_d = sps.find(g.cell_faces[faces[IA], :])
    I = np.argsort(fi)
    sgn_d = sgn_d[I]
    sgn_d = sgn_d[IC]
    sgn_d = np.ravel([sgn_d] * g.dim, "F")

    sgn = sps.diags(sgn_d, 0)

    return sgn
