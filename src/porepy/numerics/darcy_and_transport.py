from porepy.numerics.time_stepper import AbstractSolver
import numpy as np
import logging
import scipy.sparse as sps
from porepy.numerics.mixed_dim import condensation as SC

logger = logging.getLogger(__name__)

class DarcyAndTransport():
    """
    Wrapper for a stationary Darcy problem and a transport problem
    on the resulting fluxes.
    The flow and transport inputs should be members of the
    Darcy and Parabolic classes, respectively.
    A common application is solving the flow problem once, with subsequent
    stepping for the transport on a static flow field. We provide the
    convenient static_IE (implicit Euler for static flow field) option to
    greatly reduce computational time.

    """

    def __init__(self, flow, transport):
        self.flow = flow
        self.transport = transport
        if not hasattr(self.flow, 'el'):
            self.flow.el = False

    def solve(self):
        """
        Solve both problems.
        """
        p = self.flow.step()
        self.flow.pressure()
        if self.flow.el:
            SC.compute_elimination_fluxes(self.flow.full_grid, self.flow.grid(), self.flow.el_data)
        self.flow.discharge()
        s = self.transport.solve()
        return p, s[self.transport.physics]

    def save(self, export_every=1):
        """
        Save for visualization.
        """
        self.flow.save(variables=[self.flow.pressure_name])
        self.transport.save([self.transport.physics], save_every=export_every)


class static_flow_IE_solver(AbstractSolver):
     """
     Implicit time discretization:
     (y_k+1 - y_k) / dt = F^k+1
     No diffusion and static flow field is assumed. This is an adjusted
     version of the IE_solver in the time stepping module.
     """

     def __init__(self, problem):
        AbstractSolver.__init__(self, problem)

     def assemble(self):
        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_time, rhs_time = self._discretize(self.time_disc)

        self.lhs = lhs_time + lhs_flux
        self.lhs_time = lhs_time
        self.static_rhs = rhs_flux + rhs_time
        self.rhs = lhs_time * self.p0 + rhs_flux + rhs_time

     def solve(self):
        """
        Solve problem.
        """
        nt = np.ceil(self.T / self.dt).astype(np.int)
        logger.info('Time stepping using ' + str(nt) + ' steps')
        t = self.dt
        counter = 1
        self.assemble()
        IE_solver = sps.linalg.factorized((self.lhs).tocsc())
        while t < self.T + 1e-14:
            logger.info('Step ' + str(counter) + ' out of ' + str(nt))
            counter += 1
            self.update(t)
            self.step(IE_solver)
            logger.debug('Maximum value ' + str(self.p.max()) +\
                         ', minimum value ' + str(self.p.min()))
            t += self.dt
        self.update(t)
        return self.data

     def step(self, IE_solver):
          self.p = IE_solver(self.lhs_time * self.p0 + self.static_rhs)
          return self.p

     def update(self, t):
        """
        update parameters for next time step
        """
        self.p0 = self.p
        # Store result
        if self.parameters['store_results'] == True:
            self.data[self.problem.physics].append(self.p)
            self.data['times'].append(t - self.dt)
