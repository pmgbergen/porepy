import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import time
import numpy as np

from fvdiscr import mpfa
from fvdiscr import mpsa
from fvdiscr import fvutils
from fvdiscr import time_of_flight
from core.constit import second_order_tensor, fourth_order_tensor
from core.bc import bc
from core.grids import structured


class BiotDiscr(object):

    def __init__(self, grid, perm, stiffness, poro, bound_flow=None,
                 bound_mech=None, water_compr=1, water_viscosity=1, verbose=0,
                inverter='cython', eta=0):

        self.g = grid
        self.perm = perm
        self.stiffness = stiffness
        self.poro = poro


        # Boundaries and boundary conditions
        bound_faces = grid.get_boundary_faces()

        if bound_flow is None:
            self.bound_flow = bc.BoundaryCondition(grid, bound_faces.ravel('F'),
                                          ['neu'] * bound_faces.size)
        elif isinstance(bound_flow, str) \
                and (bound_flow.lower().strip() == 'dir'
                     or bound_flow.lower().strip() == 'dirichlet'):
            self.bound_flow = bc.BoundaryCondition(grid, bound_faces.ravel('F'),
                                          ['dir'] * bound_faces.size)
        elif isinstance(bound_flow, str) \
                and (bound_flow.lower().strip() == 'neu'
                     or bound_flow.lower().strip() == 'neumann'):
            self.bound_flow = bc.BoundaryCondition(grid, bound_faces.ravel('F'),
                                          ['neu'] * bound_faces.size)
        else:
            self.bound_flow = bound_flow

        if bound_mech is None:
            self.bound_mech = bc.BoundaryCondition(grid, bound_faces.ravel('F'),
                                          ['neu'] * bound_faces.size)
        elif isinstance(bound_mech, str) \
                and (bound_mech.lower().strip() == 'dir'
                     or bound_mech.lower().strip() == 'dirichlet'):
            self.bound_mech = bc.BoundaryCondition(grid, bound_faces.ravel('F'),
                                          ['dir'] * bound_faces.size)
        elif isinstance(bound_mech, str) \
                and (bound_mech.lower().strip() == 'neu'
                     or bound_mech.lower().strip() == 'neumann'):
            self.bound_mech = bc.BoundaryCondition(grid, bound_faces.ravel('F'),
                                          ['neu'] * bound_faces.size)
        else:
            self.bound_mech = bound_mech

        self.water_compr = water_compr
        self.water_viscosity = water_viscosity
        self.inverter = inverter
        self.eta = eta
        self.verbose = verbose
        self.biot_alpha = 1

    def discretize(self):
        self.discretize_flow()
        self.discretize_mech()
        self.discretize_compr()
        self.form_matrix()


    def discretize_flow(self):
        # Discretiztaion of MPFA
        tm = time.time()
        flux, bound_flux = mpfa.mpfa(self.g, self.perm, self.bound_flow,
                                     eta=self.eta, inverter=self.inverter)
        if self.verbose > 0:
            print('Time spent on mpfa discretization ' + str(time.time() - tm))
        self.flux = flux
        self.bound_flux = bound_flux


    def discretize_mech(self):
        # Discretization of elasticity / poro-mechanics
        tm = time.time()
        stress, bound_stress, grad_p, div_d, \
            stabilization = mpsa.biot(self.g, self.stiffness, self.bound_mech,
                                      eta=self.eta, inverter=self.inverter)
        if self.verbose > 0:
            print('Time spent on mpsa discretization ' + str(time.time() - tm))

        self.stress = stress
        self.bound_stress = bound_stress
        self.grad_p = grad_p
        self.div_d = div_d
        self.stabilization = stabilization

    def discretize_compr(self):
        self.compr = sps.dia_matrix((self.g.cell_volumes * self.water_compr,
                                     0),
                                    shape=(self.g.num_cells, self.g.num_cells))

    def form_matrix(self):

        div_flow = fvutils.scalar_divergence(self.g)

        # Discretization of elasticity / poro-mechanics

        div_mech = fvutils.vector_divergence(self.g)

        # Put together linear system
        A_flow = div_flow * self.flux / self.water_viscosity
        A_mech = div_mech * self.stress

        # Matrix for left hand side
        self.A_biot = sps.bmat([[A_mech, self.grad_p * self.biot_alpha],
                                [self.div_d * self.biot_alpha,
                                 self.compr + A_flow + self.stabilization]]).tocsr()


        # Matrix for right hand side (for time derivative)
        zr = np.zeros(1)
        nd = self.g.dim
        nc = self.g.num_cells
        rhs_mat_mech = sps.coo_matrix((zr, (zr, zr)),
                                      shape=(nd * nc, (nd + 1) * nc))
        rhs_mat_flow = sps.hstack([self.biot_alpha * self.div_d,
                                   self.compr + self.stabilization])
        self.rhs_mat = sps.vstack([rhs_mat_mech, rhs_mat_flow])


    def time_step(self, dt, rhs, prev_state):
        """
        Take a simple time step
        """
        b = self.rhs_mat * prev_state + dt * rhs

        # For the moment, we only use a sparse linear solver here. In the
        # future, more fance solvers and splitting methods should be included.
        du = spsolve(self.A_biot, b)

        return du

    def split_vars(self, u):
        nc = self.g.num_cells
        nd = self.g.dim
        if nd == 2:
            u_x = u[ : nd*nc : nd]
            u_y = u[1: nd*nc : nd]
            p = u[nd*nc:]
            return u_x, u_y, p
        elif nd == 3:
            u_x = u[ : nd*nc : nd]
            u_y = u[1: nd*nc : nd]
            u_z = u[2: nd*nc : nd]
            p = u[nd*nc:]
            return u_x, u_y, u_z, p
            


    def solve_time_problem(self, end_time, dt, rhs, init_state):

        def adjust_timestep(tm):
            return min(dt, end_time - tm)

        t = 0
        state = init_state
        while t < end_time:
            dt = adjust_timestep(t)
            state += self.time_step(dt, rhs, state)
            t += dt
        return state


    def stress_and_flux(self, displ, pr):
        fl = self.flux * pr
        stress = self.stress * displ
        return stress, fl
