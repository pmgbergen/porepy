import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import time
import numpy as np

from porepy.numerics.fv import mpfa, mpsa, fvutils, time_of_flight
from porepy.params import second_order_tensor, fourth_order_tensor, bc
from porepy.grids import structured
from porepy.numerics.mixed_dim.solver import Solver

class BiotDiscr(Solver):

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

        div_d = np.squeeze(data['biot_alpha'] * data['div_d'] * d)
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

        # Matrix for left hand side
        A_biot = sps.bmat([[A_mech,
                            data['grad_p'] * data['biot_alpha']],
                            [data['div_d'] * data['biot_alpha'],
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
            stabilization = mpsa.biot(g, data['stiffness'], data['bound_mech'],
                                      **data)

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



