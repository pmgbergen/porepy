import numpy as np
import logging
import time

from porepy.grids.grid_bucket import GridBucket
from porepy.numerics.linalg.linsolve import Factory as LSFactory

logger = logging.getLogger(__name__)

def color(g):
    val = np.zeros(g.num_cells, dtype=np.int)
    x = g.cell_centers[0, :]
    y = g.cell_centers[1, :]
    z = g.cell_centers[2, :]

    val[np.logical_and.reduce((x>.5, y<.5, z<.5))] = 1
    val[np.logical_and.reduce((x<.5, y>.5, z<.5))] = 2
    val[np.logical_and.reduce((x>.5, y>.5, z<.5))] = 3
    val[np.logical_and.reduce((x<.5, y<.5, z>.5))] = 4
    val[np.logical_and.reduce((x>.5, y<.5, z>.5))] = 5
    val[np.logical_and.reduce((x<.5, y>.5, z>.5))] = 6

    val[np.logical_and.reduce((x>.75, y>.75, z>.75))] = 7
    val[np.logical_and.reduce((x>.75, y>.5, y<.75, z>.75))] = 8
    val[np.logical_and.reduce((x>.5, x<.75, y>.75, z>.75))] = 9
    val[np.logical_and.reduce((x>.5, x<.75, y>.5, y<.75, z>.75))] = 10
    val[np.logical_and.reduce((x>.75, y>.75, z>.5, z<.75))] = 11
    val[np.logical_and.reduce((x>.75, y>.5, y<.75, z>.5, z<.75))] = 12
    val[np.logical_and.reduce((x>.5, x<.75, y>.75, z>.5, z<.75))] = 13

    val[np.logical_and.reduce((x>.5, x<.625, y>.5, y<.625, z>.5, z<.625))] = 14
    val[np.logical_and.reduce((x>.625, x<.75, y>.5, y<.625, z>.5, z<.625))] = 15
    val[np.logical_and.reduce((x>.5, x<.625, y>.625, y<.75, z>.5, z<.625))] = 16
    val[np.logical_and.reduce((x>.625, x<.75, y>.625, y<.75, z>.5, z<.625))] = 17
    val[np.logical_and.reduce((x>.5, x<.625, y>.5, y<.625, z>.625, z<.75))] = 18
    val[np.logical_and.reduce((x>.625, x<.75, y>.5, y<.625, z>.625, z<.75))] = 19
    val[np.logical_and.reduce((x>.5, x<.625, y>.625, y<.75, z>.625, z<.75))] = 20
    val[np.logical_and.reduce((x>.625, x<.75, y>.625, y<.75, z>.625, z<.75))] = 21

    return val

class AbstractSolver(object):
    """
    Abstract base class for solving a general first order time pde problem.
    dT/dt + G(T) = 0,
    where G(T) is a space discretization
    """

    def __init__(self, problem):
        """
        Parameters:
        ----------
        problem: a problem class. Must have the attributes
            problem.grid()
            problem.data()
            problem.space_disc()
            problem.time_disc()
            problem.time_step()
            problem.end_time()
            problem.initial_pressure()
        """
        # Get data
        g = problem.grid()

        data = problem.data()
        data[problem.physics] = []
        data['times'] = []

        p0 = problem.initial_condition()
        p = p0

        self.problem = problem
        self.g = g
        self.data = data
        self.dt = problem.time_step()
        self.T = problem.end_time()
        self.space_disc = problem.space_disc()
        self.time_disc = problem.time_disc()
        self.p0 = p0
        self.p = p
        # First initial empty lhs and rhs, then initialize them through
        # reassemble
        self.lhs = []
        self.rhs = []

    def solve(self, save_as=None, save_every=1):
        """
        Solve problem.
        """
        nt = np.ceil(self.T / self.dt).astype(np.int)
        logger.warning('Time stepping using ' + str(nt) + ' steps')
        t = self.dt
        counter = 0
        if not save_as is None:
            self.problem.split(save_as)
            self.problem.exporter.write_vtk([save_as], time_step=counter)
            times = [0.0]

        while t < self.T *(1 + 1e-14):
            logger.warning('Step ' + str(counter) + ' out of ' + str(nt))
            counter += 1
            self.update(t)
            self.reassemble()
            self.step()
            logger.debug('Maximum value ' + str(self.p.max()) +\
                         ', minimum value ' + str(self.p.min()))
            if not save_as is None and np.mod(counter, save_every)==0:
                logger.info('Saving solution')
                self.problem.split(save_as)

                #########
#                self.g.add_node_prop('color')
#                for g, d in self.g:
#                    if g.dim < 3:
#                        d['color'] = -np.ones(g.num_cells)
#                    if g.dim == 3:
#                        d['color'] = color(g)
#                        for c in np.arange(d['color'].max()+1):
#                            mask = d['color'] == c
#                            av_conc = np.average(d[save_as][mask])
#                            with open("UiB_VEM_c_"+str(c)+".csv", 'a') as f:
#                                f.write(str(t)+", "+str(av_conc)+"\n")

                #self.problem.exporter.write_vtk(['color'], time_step=counter)
                ########
                self.problem.exporter.write_vtk([save_as], time_step=counter)
                times.append(t)
                logger.info('Finished saving')
            t += self.dt

        if not save_as is None:
            self.problem.exporter.write_pvd(np.asarray(times))

        return self.data

    def step(self):
        """
        Take one time step
        """
        ls = LSFactory()
        self.p = ls.direct(self.lhs, self.rhs)
        return self.p

    def update(self, t):
        """
        update parameters for next time step
        """
        self.problem.update(t)
        self.p0 = self.p

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
        if t > self.dt + 1e-6:
            self.flag_first = False
        else:
            self.flag_first = True
        self.p_1 = self.p0
        AbstractSolver.update(self, t)

    def reassemble(self):

        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_time, rhs_time = self._discretize(self.time_disc)

        if self.flag_first:
            self.lhs = lhs_time + lhs_flux
            self.rhs = lhs_time * self.p0 + rhs_flux + rhs_time
        else:
            self.lhs = lhs_time + 2. / 3 * lhs_flux
            bdf2_rhs = 4. / 3 * lhs_time * self.p0 - 1. / 3 * lhs_time * self.p_1
            self.rhs = bdf2_rhs + 2. / 3 * rhs_flux + rhs_time


class Explicit(AbstractSolver):
    """
    Explicit time discretization:
    (y_k - y_k-1)/dt = F^k
    """

    def __init__(self, problem):
        AbstractSolver.__init__(self, problem)

    def solve(self, save_as=None, save_every=1):
        """
        Solve problem.
        """
        t = 0.0
        counter = 0
        # Save initial condition
        if not save_as is None:
            self.problem.split(save_as)
            self.problem.exporter.write_vtk([save_as], time_step=counter)
            times = [0.0]

        while t < self.T - self.dt + 1e-14:
            self.update(t)
            self.reassemble()
            self.step()
            # Save time step
            if not save_as is None and np.mod(counter, save_every)==0:
                logger.info('Saving solution')
                self.problem.split(save_as)
                self.problem.exporter.write_vtk([save_as], time_step=counter)
                times.append(t)
                logger.info('Finished saving')
            t += self.dt

        # Write pvd
        if not save_as is None:
            self.problem.exporter.write_pvd(np.asarray(times))

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
