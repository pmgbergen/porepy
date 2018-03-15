import numpy as np
import logging
import time

import porepy as pp

logger = logging.getLogger(__name__)


class AbstractSolver(object):
    """
    Abstract base class for solving a general first order time pde problem.
    dT/dt + G(T) = 0,
    where G(T) is a space discretization
    """

    def __init__(self, model, dt=1., end_time=1.):
        """
        Parameters:
        ----------
        model: a model class. Must have the attributes
            model.grid()
            model.data()
            model.space_disc()
            model.mass_disc()
            model.initial_condition()
        """
        # Get data
        self.model = model
        self.gb = model.grid()
        self.data = model.data()
        self.dt = dt
        self.T = end_time
        self.space_disc = model.space_disc()
        self.mass_disc = model.mass_disc()
        self.x0 = model.initial_condition()
        self.x = self.x0
        # First initial empty lhs and rhs, then initialize them through
        # assemble
        self.lhs = []
        self.rhs = []
        self.rhs_var = []
        self.rhs_const = []

    def solve(self, file_name=None, var_name='solution', save_every=1):
        """
        Solve problem.
        """
        nt = np.ceil(self.T / self.dt).astype(np.int)
        logger.warning('Time stepping using ' + str(nt) + ' steps')
        t = self.dt
        counter = 0
        if not file_name is None:
            exporter = pp.Exporter(self.gb, file_name)
            self.model.split(self.x, var_name)
            self.model.exporter.write_vtk([var_name], time_step=counter)
            times = [0.0]

        while t < self.T *(1 + 1e-14):
            logger.warning('Step ' + str(counter) + ' out of ' + str(nt))
            counter += 1
            self.assemble()
            self.update()
            self.step()
            logger.debug('Maximum value ' + str(self.x.max()) +\
                         ', minimum value ' + str(self.x.min()))
            if not file_name is None and np.mod(counter, save_every)==0:
                logger.info('Saving solution')
                self.model.split(self.x, var_name)
                exporter.write_vtk([var_name], time_step=counter)
                times.append(t)
                logger.info('Finished saving')
            t += self.dt

        if not file_name is None:
            exporter.write_pvd(np.asarray(times))

        return self.data

    def step(self):
        """
        Take one time step.
        """
        raise NotImplementedError(
            'subclass must overload function reasemble()')

    def update(self):
        """
        update parameters for next time step
        """
        self.x0 = self.x

    def assemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        raise NotImplementedError(
            'subclass must overload function reasemble()')

    def _discretize(self, discs):
        if isinstance(self.gb, pp.GridBucket):
            if not isinstance(discs, tuple):
                discs = [discs]
            lhs, rhs = np.array(discs[0].matrix_rhs(self.gb))
            for disc in discs[1:]:
                lhs_n, rhs_n = disc.matrix_rhs(self.gb)
                lhs += lhs_n
                rhs += rhs_n
        else:
            if not isinstance(discs, tuple):
                discs = [discs]
            lhs, rhs = discs[0].matrix_rhs(self.gb, self.data)
            for disc in discs[1:]:
                lhs_n, rhs_n = disc.matrix_rhs(self.gb, self.data)
                lhs += lhs_n
                rhs += rhs_n
        return lhs, rhs


class Implicit(AbstractSolver):
    """
    Implicit time discretization:
    (y_k+1 - y_k) / dt = F^k+1
    """
    def step(self):
        ls = pp.LSFactory()
        self.rhs = self.rhs_var * self.x0 + self.rhs_const
        self.x = ls.direct(self.lhs, self.rhs)
        return self.x

    def assemble(self):
        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_mass, rhs_mass = self._discretize(self.mass_disc)

        self.rhs_var = lhs_mass / self.dt
        self.rhs_const = rhs_mass / self.dt + rhs_flux
        self.lhs = lhs_mass / self.dt + lhs_flux


class BDF2(AbstractSolver):
    """
    Second order implicit time discretization:
    (y_k+2 - 4/3 * y_k+1 + 1/3 * y_k) / dt = 2/3 * F^k+2
    """

    def __init__(self, *vargs, **kwargs):
        AbstractSolver.__init__(self, *vargs, **kwargs)
        self.x_1 = self.x0
        self.flag_first = True
        self.did_step = False

    def step(self):
        if self.flag_first:
            bdf2_rhs = self.rhs_var * self.x0
        else:
            bdf2_rhs = 4. / 3 * self.rhs_var * self.x0 - \
                       1. / 3 * self.rhs_var * self.x_1
        self.rhs = bdf2_rhs + self.rhs_const

        ls = pp.LSFactory()
        self.x = ls.direct(self.lhs, self.rhs)
        self.did_step = True

    def update(self):
        """
        update parameters for next time step
        """
        if self.flag_first and self.did_step:
            self.flag_first = False
        self.x_1 = self.x0
        AbstractSolver.update(self)

    def assemble(self):
        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_mass, rhs_mass = self._discretize(self.mass_disc)
        self.rhs_var = lhs_mass / self.dt
        if self.flag_first:
            self.lhs = lhs_mass / self.dt + lhs_flux
            self.rhs_const = rhs_flux + rhs_mass / self.dt
        else:
            self.lhs = lhs_mass / self.dt + 2. / 3 * lhs_flux
            self.rhs_const = 2. / 3 * rhs_flux + rhs_mass / self.dt

class Explicit(AbstractSolver):
    """
    Explicit time discretization:
    (y_k - y_k-1)/dt = F^k
    """
    def step(self):
        ls = pp.LSFactory()
        self.rhs = self.rhs_var * self.x0 + self.rhs_const
        self.x = ls.direct(self.lhs, self.rhs)
        return self.x

    def assemble(self):

        lhs_flux, rhs_flux = self._discretize(self.space_disc)
        lhs_mass, rhs_mass = self._discretize(self.mass_disc)

        self.lhs = lhs_mass / self.dt
        self.rhs_var = (lhs_mass / self.dt - lhs_flux)
        self.rhs_const = rhs_flux + rhs_mass


class CrankNicolson(AbstractSolver):
    """
    Crank-Nicolson time discretization:
    (y_k+1 - y_k) / dt = 0.5 * (F^k+1 + F^k)
    """

    def __init__(self, model, **kwargs):
        AbstractSolver.__init__(self, model, **kwargs)
        self.lhs_flux, self.rhs_flux = self._discretize(model.space_disc())
        self.lhs_mass, self.rhs_mass = self._discretize(model.mass_disc())
        self.lhs_flux_0 = self.lhs_flux
        self.rhs_flux_0 = self.rhs_flux
        self.lhs_mass_0 = self.lhs_mass
        self.rhs_mass_0 = self.rhs_mass

    def step(self):
        self.rhs = self.rhs_var * self.x0 + self.rhs_const
        ls = pp.LSFactory()
        self.x = ls.direct(self.lhs, self.rhs)
        return self.x

    def update(self):
        """
        update parameters for next time step
        """
        AbstractSolver.update(self)
        self.lhs_flux_0 = self.lhs_flux
        self.rhs_flux_0 = self.rhs_flux
        self.lhs_mass_0 = self.lhs_mass
        self.rhs_mass_0 = self.rhs_mass

    def assemble(self):
        self.lhs_flux, self.rhs_flux = self._discretize(self.space_disc)
        self.lhs_mass, self.rhs_mass = self._discretize(self.mass_disc)

        self.lhs = self.lhs_mass / self.dt + 0.5 * self.lhs_flux
        self.rhs_var = self.lhs_mass / self.dt - 0.5 * self.lhs_flux_0
        rhs1 = 0.5 * (self.rhs_flux + self.rhs_mass / self.dt)
        rhs0 = 0.5 * (self.rhs_flux_0 + self.rhs_mass_0 / self.dt)
        self.rhs_const = rhs1 + rhs0
