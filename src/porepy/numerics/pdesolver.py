import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.grids.grid_bucket import GridBucket
from porepy.params.data import Parameters
from porepy.params import tensor
from porepy.params import bc
from porepy.numerics.mixed_dim.solver import Solver


class AbstractSolver():
    """
    Class for solving slightly compressible flow.
    We solve the equation:
    phi * dp/dt - nabla * K * grad(p) + v * nabla p  = f.
    """

    def __init__(self, problem):
        """
        Parameters:
        ----------
        problem: a problem class. Must have the attributes
            problem.grid()
            problem.porosity(): phi
            problem.space_disc()
            problem.diffusivity(): K
            problem.time_step()
            problem.end_time()
            problem.bc()
            problem.bc_val(t)
            problem.initial_pressure()
            problem.source(t): f
        """
        # Get data
        g = problem.grid()

        dt = problem.time_step()
        T = problem.end_time()
        data = problem.data()
        data[problem.physics] = []
        data['times'] = []

        problem.update(dt)

        space_disc = problem.space_disc()
        time_disc = problem.time_disc()

        p0 = problem.initial_condition()
        p = p0
        data[problem.physics].append(p)
        data['times'].append(0.0)

        self.problem = problem
        self.g = g
        self.data = data
        self.dt = dt
        self.T = T
        self.space_disc = space_disc
        self.time_disc = time_disc
        self.p0 = p0
        self.p = p
        # First initial empty lhs and rhs, then initialize them through
        # reassemble
        self.lhs = []
        self.rhs = []
        self.reassemble()

        self.parameters = {'store_results': False, 'verbose': False}

    def solve(self):
        """
        Solve problem.
        """
        t = self.dt
        while t < self.T + 1e-14:
            if self.parameters['verbose']:
                print('solving time step: ', t)
            self.step()
            self.update(t)
            self.reassemble()
            t += self.dt

        return self.data

    def step(self):
        """
        Take one time step
        """
        self.p = sps.linalg.spsolve(self.lhs, self.rhs)

    def update(self, t):
        """
        update parameters for next time step
        """
        self.problem.update(t + self.dt)
        self.p0 = self.p
        # Store result
        if self.parameters['store_results'] == True:
            self.data[self.problem.physics].append(self.p)
            self.data['times'].append(t)

    def reassemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        raise NotImplementedError(
            'subclass must overload function reasemble()')

    def _discretize(self, discs):
        if isinstance(self.g, GridBucket):
            if not isinstance(discs, tuple):
                discs = [discs]
            lhs, rhs = np.array(discs[0].matrix_rhs(self.g))
            for disc in discs[1:]:
                lhs_n, rhs_n = disc.matrix_rhs(self.g)
                lhs += lhs_n
                rhs += rhs_n
        else:
            if not isinstance(discs, tuple):
                discs = [discs]
            lhs, rhs = discs[0].matrix_rhs(self.g, self.data)
            for disc in discs[1:]:
                lhs_n, rhs_n = disc.matrix_rhs(self.g, self.data)
                lhs += lhs_n
                rhs += rhs_n
        return lhs, rhs


class Implicit(AbstractSolver):
    """
    Implicit time discretization:
    (y_k+1 - y_k) / dt = F^k+1
    """

    def __init__(self, problem):
        AbstractSolver.__init__(self, problem)

    def reassemble(self):

        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_time, rhs_time = self._discretize(self.time_disc)

        self.lhs = lhs_time + lhs_flux
        self.rhs = lhs_time * self.p0 + rhs_flux + rhs_time


class BDF2(AbstractSolver):
    """
    Second order implicit time discretization:
    (y_k+2 - 4/3 * y_k+1 + 1/3 * y_k) / dt = 2/3 * F^k+2
    """

    def __init__(self, problem):
        self.flag_first = True
        AbstractSolver.__init__(self, problem)
        self.p_1 = self.p0

    def update(self, t):
        """
        update parameters for next time step
        """
        self.flag_first = False
        self.p_1 = self.p0
        AbstractSolver.update(self, t)

    def reassemble(self):

        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_time, rhs_time = self._discretize(self.time_disc)

        if self.flag_first:
            self.lhs = lhs_time + lhs_flux
            self.rhs = lhs_time * self.p0 + rhs_flux + rhs_time
        else:
            self.lhs = lhs_time + 2 / 3 * lhs_flux
            bdf2_rhs = 4 / 3 * lhs_time * self.p0 - 1 / 3 * lhs_time * self.p_1
            self.rhs = bdf2_rhs + 2 / 3 * rhs_flux + rhs_time


class Explicit(AbstractSolver):
    """
    Explicit time discretization:
    (y_k - y_k-1)/dt = F^k
    """

    def __init__(self, problem):
        AbstractSolver.__init__(self, problem)

    def solve(self):
        """
        Solve problem.
        """
        t = self.dt
        while t < self.T + 1e-14:
            if self.parameters['verbose']:
                print('solving time step: ', t)
            self.step()
            self.update(t - self.dt)
            self.reassemble()
            t += self.dt
        return self.data

    def reassemble(self):

        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_time, rhs_time = self._discretize(self.time_disc)

        self.lhs = lhs_time
        self.rhs = (lhs_time - lhs_flux) * self.p0 + rhs_flux + rhs_time


class CrankNicolson(AbstractSolver):
    """
    Crank-Nicolson time discretization:
    (y_k+1 - y_k) / dt = 0.5 * (F^k+1 + F^k)
    """

    def __init__(self, problem):
        self.g = problem.grid()
        self.lhs_flux, self.rhs_flux = self._discretize(problem.space_disc())
        self.lhs_time, self.rhs_time = self._discretize(problem.time_disc())
        self.lhs_flux_0 = self.lhs_flux
        self.rhs_flux_0 = self.rhs_flux
        self.lhs_time_0 = self.lhs_time
        self.rhs_time_0 = self.rhs_time
        AbstractSolver.__init__(self, problem)

    def update(self, t):
        """
        update parameters for next time step
        """
        AbstractSolver.update(self, t)
        self.lhs_flux_0 = self.lhs_flux
        self.rhs_flux_0 = self.rhs_flux
        self.lhs_time_0 = self.lhs_time
        self.rhs_time_0 = self.rhs_time

    def reassemble(self):
        self.lhs_flux, self.rhs_flux = self._discretize(self.space_disc)
        self.lhs_time, self.rhs_time = self._discretize(self.time_disc)

        rhs1 = 0.5 * (self.rhs_flux + self.rhs_time)
        rhs0 = 0.5 * (self.rhs_flux_0 + self.rhs_time_0)
        self.lhs = self.lhs_time + 0.5 * self.lhs_flux
        self.rhs = (self.lhs_time - 0.5 * self.lhs_flux_0) * \
            self.p0 + rhs1 + rhs0
