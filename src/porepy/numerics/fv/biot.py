import scipy.sparse as sps
import scipy.sparse.linalg as la
import time
import numpy as np

from porepy.numerics.fv import mpfa, mpsa, fvutils
from porepy.params import second_order_tensor, fourth_order_tensor, bc
from porepy.numerics.mixed_dim.solver import Solver


class Biot(Solver):

    def ndof(self, g):
        """ Return the number of degrees of freedom associated wiht the method.

        In this case, each cell has nd displacement variables, as well as a
        pressure variable.

        Parameters:
            g: grid, or a subclass.

        Returns:
            int: Number of degrees of freedom in the grid.

        """
        return g.num_cells * (1 + g.dim)


    def matrix_rhs(self, g, data, discretize=True):
        if discretize:
            self.discretize(g, data)
        A_biot = self.assemble_matrix(g, data)
        rhs_bound = self.rhs_bound(g, data)
        return A_biot, rhs_bound

#--------------------------- Helper methods for discretization ----------

    def rhs_bound(self, g, data):
        """ Boundary component of the right hand side (dependency on previous
        time step).

        TODO: Boundary effects of coupling terms.

        Parameters:
            g: grid, or subclass, with geometry fields computed.
            data: dictionary to store the data terms. Must have been through a
                call to discretize() to discretization of right hand side.
            state: np.ndarray, solution vector from previous time step.

        Returns:
            np.ndarray: Contribution to right hand side given the current
            state.

        """
        d = data['bound_mech_val']
        p = data['bound_flow_val']

        div_flow = fvutils.scalar_divergence(g)
        div_mech = fvutils.vector_divergence(g)

        p_bound = div_flow * data['bound_flux'] * p
        s_bound = div_mech * data['bound_stress'] * d
        return np.hstack((s_bound, p_bound))

    def rhs_time(self, g, data, state):
        """ Time component of the right hand side (dependency on previous time
        step).

        TODO: 1) Generalize this to allow other methods than Euler backwards?
              2) How about time dependent boundary conditions.

        Parameters:
            g: grid, or subclass, with geometry fields computed.
            data: dictionary to store the data terms. Must have been through a
                call to discretize() to discretization of right hand side.
            state: np.ndarray, solution vector from previous time step.

        Returns:
            np.ndarray: Contribution to right hand side given the current
            state.

        """
        d = self.extractD(g, state, as_vector=True)
        p = self.extractP(g, state)

        d_scaling = data.get('displacement_scaling', 1)

        div_d = np.squeeze(data['biot_alpha'] * data['div_d'] * d * d_scaling)
        p_cmpr = data['compr_discr'] * p

        mech_rhs = np.zeros(g.dim * g.num_cells)

        return np.hstack((mech_rhs, div_d + p_cmpr))


    def discretize(self, g, data):
        """ Discretize flow and mechanics equations using FV methods.

        """
        # Discretization of elasticity / poro-mechanics
        self._discretize_flow(g, data)
        self._discretize_mech(g, data)
        self._discretize_compr(g, data)


    def assemble_matrix(self, g, data):
        div_flow = fvutils.scalar_divergence(g)
        div_mech = fvutils.vector_divergence(g)

        # Put together linear system
        A_flow = div_flow * data['flux'] / data['water_viscosity']
        A_mech = div_mech * data['stress']

        # Time step size
        dt = data['dt']

        d_scaling = data.get('displacement_scaling', 1)
        # Matrix for left hand side
        A_biot = sps.bmat([[A_mech,
                            data['grad_p'] * data['biot_alpha']],
                            [data['div_d'] * data['biot_alpha'] * d_scaling,
                             data['compr_discr'] \
                             + dt * A_flow + data['stabilization']]]).tocsr()

        return A_biot


    def _discretize_flow(self, g, data):

        perm = data.get('perm')
        bound_flow = data.get('bound_flow')
        # Discretiztaion of MPFA
        flux, bound_flux = mpfa.mpfa(g, perm, bound_flow, **data)
        data['flux'] = flux
        data['bound_flux'] = bound_flux


    def _discretize_mech(self, g, data):
        # Discretization of elasticity / poro-mechanics
        stress, bound_stress, grad_p, div_d, \
            stabilization = self.discretize(g, data['stiffness'],
                                            data['bound_mech'], **data)

        data['stress'] = stress
        data['bound_stress'] = bound_stress
        data['grad_p'] = grad_p
        data['div_d'] = div_d
        data['stabilization'] = stabilization

    def _discretize_compr(self, g, data):
        compr = data.get('fluid_compr', 0)
        poro = data['poro']
        data['compr_discr'] = sps.dia_matrix((g.cell_volumes * compr * poro, 0),
                                             shape=(g.num_cells, g.num_cells))


#----------------------- Linear solvers -------------------------------------

    def solve(self, A, solver='direct', **kwargs):

        solver = solver.strip().lower()
        if solver == 'direct':
            def slv(b):
                x = la.spsolve(A, b)
                return x
        elif solver == 'factorized':
            slv = la.factorized(A.to_csc())

        else:
            raise ValueError('Unknown solver ' + solver)

        return slv


#----------------------- Methods for post processing -------------------------
    def extractD(self, g, u, dims=None, as_vector=False):
        """ Extract displacement field from solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.
            dim (list of int, optional): Which dimension to extract. If None,
                all dimensions are returned.
        Returns:
            list of np.ndarray: Displacement variables in the specified
                dimensions.

        """
        if dims is None:
            dims = np.arange(g.dim)
        vals = []

        inds = np.arange(0, g.num_cells * g.dim, g.dim)

        for d in dims:
            vals.append(u[d + inds])
        if as_vector:
            vals = np.asarray(vals).reshape((-1, 1), order='C')
            return vals
        else:
            return vals


    def extractP(self, g, u):
        """ Extract pressure field from solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.

        Returns:
            np.ndarray: Pressure part of solution vector.

        """
        return u[g.dim * g.num_cells:]


    def compute_flux(self, g, u, data):
        """ Compute flux field corresponding to a solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.
            bc_flow (np.ndarray): Flux boundary values.
            data (dictionary): Dictionary related to grid and problem. Should
                contain boundary discretization.

        Returns:
            np.ndarray: Flux over all faces

        """
        flux_discr = data['flux']
        bound_flux = data['bound_flux']
        bound_val = data['bound_flow_val']
        p = self.extractP(g, u)
        flux = flux_discr * p + bound_flux * bound_val
        return flux

    def compute_stress(self, g, u, data):
        """ Compute stress field corresponding to a solution.

        Parameters:
            g: grid, or a subclass.
            u (np.ndarray): Solution variable, representing displacements and
                pressure.
            bc_flow (np.ndarray): Flux boundary values.
            data (dictionary): Dictionary related to grid and problem. Should
                contain boundary discretization.

        Returns:
            np.ndarray, g.dim * g.num_faces: Stress over all faces. Stored as
                all stress values on the first face, then the second etc.

        """
        stress_discr = data['stress']
        bound_stress = data['bound_stress']
        bound_val = data['bound_mech_val']
        d = self.extractD(g, u, as_vector=True)
        stress = stress_discr * d + (bound_stress * bound_val)[:, np.newaxis]
        return stress

#-------------------------------------------------------------------------

    def discretize(self, g, constit, bound, faces=None, eta=0, inverter=None,
                   **kwargs):
        """
        Discretization of poro-elasticity by the MPSA-W method.

        Implementation needs (in addition to those mentioned in mpsa function):
            1) Fields for non-zero boundary conditions. Should be simple.
            2) Split return value grad_p into forces and a divergence operator,
            so that we can compute Biot forces on a face.

        Parameters:
            g (core.grids.grid): grid to be discretized
            k (core.constit.second_order_tensor) permeability tensor
            constit (core.bc.bc) class for boundary values
            faces (np.ndarray) faces to be considered. Intended for partial
                discretization, may change in the future
            eta Location of pressure continuity point. Should be 1/3 for simplex
                grids, 0 otherwise. On boundary faces with Dirichlet conditions,
                eta=0 will be enforced.
            inverter (string) Block inverter to be used, either numba (default),
                cython or python. See fvutils.invert_diagonal_blocks for details.

        Returns:
            scipy.sparse.csr_matrix (shape num_faces * dim, num_cells * dim): stres
                discretization, in the form of mapping from cell displacement to
                face stresses.
            scipy.sparse.csr_matrix (shape num_faces * dim, num_faces * dim):
                discretization of boundary conditions. Interpreted as istresses
                induced by the boundary condition (both Dirichlet and Neumann). For
                Neumann, this will be the prescribed stress over the boundary face,
                and possibly stress on faces having nodes on the boundary. For
                Dirichlet, the values will be stresses induced by the prescribed
                displacement.  Incorporation as a right hand side in linear system
                by multiplication with divergence operator.
            scipy.sparse.csr_matrix (shape num_cells * dim, num_cells): Forces from
                the pressure gradient (I*p-term), represented as body forces.
                TODO: Should rather be represented as forces on faces.
            scipy.sparse.csr_matrix (shape num_cells, num_cells * dim): Trace of
                strain matrix, cell-wise.
            scipy.sparse.csr_matrix (shape num_cells x num_cells): Stabilization
                term.

        Example:
            # Set up a Cartesian grid
            g = structured.CartGrid([5, 5])
            c = fourth_order_tensor.FourthOrderTensor(g.dim, np.ones(g.num_cells))
            k = second_order_tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))

            # Dirirchlet boundary conditions for mechanics
            bound_faces = g.get_boundary_faces().ravel()
            bnd = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)

            # Use no boundary conditions for flow, will default to homogeneous
            # Neumann.

            # Discretization
            stress, bound_stress, grad_p, div_d, stabilization = biot(g, c, bnd)
            flux, bound_flux = mpfa(g, k, None)

            # Source in the middle of the domain
            q_mech = np.zeros(g.num_cells * g.dim)

            # Divergence operator for the grid
            div_mech = fvutils.vector_divergence(g)
            div_flow = fvutils.scalar_divergence(g)
            a_mech = div_mech * stress
            a_flow = div_flow * flux

            a_biot = sps.bmat([[a_mech, grad_p], [div_d, a_flow +
                                                           stabilization]])

            # Zero boundary conditions by default.

            # Injection in the middle of the domain
            rhs = np.zeros(g.num_cells * (g.dim + 1))
            rhs[g.num_cells * g.dim + np.ceil(g.num_cells / 2)] = 1
            x = sps.linalg.spsolve(A, rhs)

            u_x = x[0:g.num_cells * g.dim: g.dim]
            u_y = x[1:g.num_cells * g.dim: g.dim]
            p = x[g.num_cells * gdim:]

        """

        # The grid coordinates are always three-dimensional, even if the grid
        # is really 2D. This means that there is not a 1-1 relation between the
        # number of coordinates of a point / vector and the real dimension.
        # This again violates some assumptions tacitly made in the
        # discretization (in particular that the number of faces of a cell that
        # meets in a vertex equals the grid dimension, and that this can be
        # used to construct an index of local variables in the discretization).
        # These issues should be possible to overcome, but for the moment, we
        # simply force 2D grids to be proper 2D.
        if g.dim == 2:
            g = g.copy()
            g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
            g.face_centers = np.delete(g.face_centers, (2), axis=0)
            g.face_normals = np.delete(g.face_normals, (2), axis=0)
            g.nodes = np.delete(g.nodes, (2), axis=0)

            constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=0)
            constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=1)
        nd = g.dim

        # Define subcell topology
        subcell_topology = fvutils.SubcellTopology(g)
        # Obtain mappings to exclude boundary faces
        bound_exclusion = fvutils.ExcludeBoundaries(subcell_topology, bound, nd)

        num_subhfno = subcell_topology.subhfno.size

        num_nodes = np.diff(g.face_nodes.indptr)
        sgn = g.cell_faces[subcell_topology.fno, subcell_topology.cno].A

        def build_rhs_normals_single_dimension(dim):
            val = g.face_normals[dim, subcell_topology.fno] \
                * sgn / num_nodes[subcell_topology.fno]
            mat = sps.coo_matrix((val.squeeze(), (subcell_topology.subfno,
                                                  subcell_topology.cno)),
                                 shape=(subcell_topology.num_subfno,
                                        subcell_topology.num_cno))
            return mat

        rhs_normals = build_rhs_normals_single_dimension(0)
        for iter1 in range(1, nd):
            this_dim = build_rhs_normals_single_dimension(iter1)
            rhs_normals = sps.vstack([rhs_normals, this_dim])

        rhs_normals = bound_exclusion.exclude_dirichlet_nd(rhs_normals)

        num_dir_subface = (bound_exclusion.exclude_neu.shape[1] -
                           bound_exclusion.exclude_neu.shape[0]) * nd
        rhs_normals_displ_var = sps.coo_matrix((nd * subcell_topology.num_subfno
                                                - num_dir_subface,
                                                subcell_topology.num_cno))

        # Why minus?
        rhs_normals = -sps.vstack([rhs_normals, rhs_normals_displ_var])
        del rhs_normals_displ_var

        # Call core part of MPSA
        hook, igrad, rhs_cells, cell_node_blocks, hook_normal \
            = mpsa.mpsa_elasticity(g, constit, subcell_topology, bound_exclusion,
                                   eta, inverter)

        # Output should be on face-level (not sub-face)
        hf2f = _map_hf_2_f(subcell_topology.fno_unique,
                           subcell_topology.subfno_unique, nd)

        # Stress discretization
        stress = hf2f * hook * igrad * rhs_cells

        # Right hand side for boundary discretization
        rhs_bound = mpfa.create_bound_rhs(bound, bound_exclusion, subcell_topology, g)
        # Discretization of boundary values
        bound_stress = hf2f * hook * igrad * rhs_bound

        del hook, rhs_bound

        # Face-wise gradient operator. Used for the term grad_p in Biot's
        # equations.
        rows = fvutils.expand_indices_nd(subcell_topology.cno, nd)
        cols = np.arange(num_subhfno * nd)
        vals = np.tile(sgn, (nd, 1)).ravel('F')
        div_gradp = sps.coo_matrix((vals, (rows, cols)),
                                   shape=(subcell_topology.num_cno * nd,
                                          num_subhfno * nd)).tocsr()

        del rows, cols, vals

        # Normal vectors, used for computing pressure gradient terms in
        # Biot's equations. These are mappings from cells to their faces,
        # and are most easily computed prior to elimination of subfaces (below)
        # ind_face = np.argsort(np.tile(subcell_topology.subhfno, nd))
        # hook_normal = sps.coo_matrix((np.ones(num_subhfno * nd),
        #                               (np.arange(num_subhfno*nd), ind_face)),
        # shape=(nd*num_subhfno, ind_face.size)).tocsr()

        grad_p = div_gradp * hook_normal * igrad * rhs_normals
        # assert np.allclose(grad_p.sum(axis=0), np.zeros(g.num_cells))

        del hook_normal, div_gradp

        num_cell_nodes = g.num_cell_nodes()
        cell_vol = g.cell_volumes / num_cell_nodes

        if nd == 2:
            trace = np.array([0, 3])
        elif nd == 3:
            trace = np.array([0, 4, 8])
        row, col = np.meshgrid(np.arange(cell_node_blocks.shape[1]), trace)
        incr = np.cumsum(nd**2 * np.ones(cell_node_blocks.shape[1])) - nd**2
        col += incr.astype('int32')
        val = np.tile(cell_vol[cell_node_blocks[0]], (nd, 1))
        vector_2_scalar = sps.coo_matrix((val.ravel('F'),
                                          (row.ravel('F'),
                                           col.ravel('F')))).tocsr()
        del row, col, val
        div_op = sps.coo_matrix((np.ones(cell_node_blocks.shape[1]),
                                 (cell_node_blocks[0], np.arange(
                                     cell_node_blocks.shape[1])))).tocsr()
        div = div_op * vector_2_scalar
        del div_op, vector_2_scalar

        div_d = div * igrad * rhs_cells
        del rhs_cells

        stabilization = div * igrad * rhs_normals

        return stress, bound_stress, grad_p, div_d, stabilization
