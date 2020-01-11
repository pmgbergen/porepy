#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functionality for linear solvers.

Created on Thu Nov  2 19:08:28 2017

@author: Eirik Keilegavlen
"""
import numpy as np
import scipy.sparse.linalg as spl
import logging

logger = logging.getLogger(__name__)

try:
    import pyamg
except ImportError:
    logger.info(
        " Could not import the pyamg package. pyamg solver will not be available."
    )


class IterCounter(object):
    """ Simple callback function for iterative solvers.

    Taken from https://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method
    """

    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            logger.info("iter %3i\trk = %s" % (self.niter, str(rk)))


class Factory:
    """ Factory class for linear solver functionality. The intention is to
    provide a single entry point for all relevant linear solvers. Hopefully,
    this will pay off as the number of options expands.

    Currently supported backends are standard scipy.sparse solvers, and pyamg.
    For information on parameters etc, confer the wrapped functions and
    libraries.

    """

    def ilu(self, A, **kwargs):
        """ Wrapper around ILU function from scipy.sparse.linalg.
        Confer that function for documetnation.

        Parameters:
            A: Matrix to be factorized
            **kwargs: Parameters passed on to scipy.sparse

        Returns:
            scipy.sparse.LinearOperator: Ready to be used as a preconditioner.

        """
        opts = self.__extract_spilu_args(**kwargs)
        iA = spl.spilu(A, **opts)
        iA_x = lambda x: iA.solve(x)
        return spl.LinearOperator(A.shape, iA_x)

    def lu(self, A, **kwargs):
        """ Wrapper around LU function from scipy.sparse.linalg.
        Confer that function for documetnation.

        Parameters:
            A: Matrix to be factorized
            **kwargs: Parameters passed on to scipy.sparse

        Returns:
            Solver function from LU.

        """
        opts = self.__extract_splu_args(**kwargs)
        iA = spl.splu(A, **opts)
        return iA.solve

    def direct(self, A, rhs=None):
        """ Wrapper around spsolve from scipy.sparse.linalg.
        Confer that function for documetnation.

        Parameters:
            A: Matrix to be factorized
            rhs (optional): Right hand side vector. If not provided, a funciton
                to solve with the given A is returned instead.

        Returns:
            Either scipy.sparse.LinearOperator: Ready to be used as a
                preconditioner, or the solution A^-1 b

        """

        def solve(b):
            return spl.spsolve(A, b)

        if rhs is None:
            return solve
        else:
            return solve(rhs)

    def gmres(self, A):
        """ Wrapper around gmres function from scipy.sparse.linalg.
        Confer that function for documetnation.

        Parameters:
            A (Matrix): Left hand side matrix

        Returns:
            A function that wraps gmres. The function needs a right hand side
                vector of appropriate size. Also accepts further keyword
                arguments that are passed on to scipy.

        """

        def solve(b, **kwargs):
            opt = self.__extract_gmres_args(**kwargs)
            return spl.gmres(A, b, **opt)

        return solve

    def cg(self, A):
        """ Wrapper around cg function from scipy.sparse.linalg.
        Confer that function for documetnation.

        Parameters:
            A (Matrix): Left hand side matrix

        Returns:
            A function that wraps cg. The function needs a right hand side
                vector of appropriate size. Also accepts further keyword
                arguments that are passed on to scipy.

        """

        def solve(b, **kwargs):
            opt = self.__extract_krylov_args(**kwargs)
            return spl.cg(A, b, **opt)

        return solve

    def bicgstab(self, A, **kwargs):
        """ Wrapper around bicgstab function from scipy.sparse.linalg.
        Confer that function for documetnation.

        Parameters:
            A (Matrix): Left hand side matrix

        Returns:
            A function that wraps bicgstab. The function needs a right hand
                side vector of appropriate size. Also accepts further keyword
                arguments that are passed on to scipy.

        """

        def solve(b, **kwargs):
            opt = self.__extract_krylov_args(**kwargs)
            return spl.bicgstab(A, b, **opt)

        return solve

    def amg(self, A, null_space=None, as_precond=True, **kwargs):
        """ Wrapper around the pyamg solver by Bell, Olson and Schroder.

        For the moment, the method creates a smoothed aggregation amg solver.
        More elaborate options may be added in the future if the need arises.
        If you need other types of solvers or functionality, access pyamg
        directly.

        For documentation of pyamg, including parameters options, confer
        https://github.com/pyamg/pyamg.

        This wrapper can either produce a solver or a preconditioner
        (LinearOperator). For the moment we provide limited parsing of options,
        the solver will be a GMRES accelerated V-cycle, while the
        preconditioner is simply a V-cycle. Expanding this is not difficult,
        but tedious.

        Parameters:
            A (Matrix): To be factorized.
            null_space (optional): Null space of the matrix. Accurate
                information here is essential to creating a good coarse space
                hierarchy. Defaults to vector of ones, which is the correct
                choice for standard elliptic equations.
            as_precond (optional, defaults to True): Whether to return a solver
                or a preconditioner function.
            **kwargs: For the moment not in use.

        Returns:
            Function: Either a LinearOperator to be used as preconditioner,
                or a solver.

        """

        if null_space is None:
            null_space = np.ones(A.shape[0])
        try:
            ml = pyamg.smoothed_aggregation_solver(A, B=null_space)
        except NameError:
            raise ImportError(
                "Using amg needs requires the pyamg package. pyamg was not imported"
            )

        def solve(b, res=None, **kwargs):
            if res is None:
                return ml.solve(b, accel="gmres", cycle="V")
            else:
                return ml.solve(b, residuals=res, accel="gmres", cycle="V")

        if as_precond:
            M_x = lambda x: ml.solve(x, tol=1e-20, maxiter=10, cycle="W")
            return spl.LinearOperator(A.shape, M_x)
        else:
            return solve

    #### Helper functions below

    def __extract_krylov_args(self, **kwargs):
        d = {}
        d["x0"] = kwargs.get("x0", None)
        d["tol"] = kwargs.get("tol", 1e-5)
        d["maxiter"] = kwargs.get("maxiter", None)
        d["M"] = kwargs.get("M", None)
        cb = kwargs.get("callback", None)
        if cb is not None and not isinstance(cb, IterCounter):
            cb = IterCounter()
        d["callback"] = cb
        return d

    def __extract_gmres_args(self, **kwargs):
        d = self.__extract_krylov_args(**kwargs)
        d["restart"] = kwargs.get("restart", None)
        return d

    def __extract_splu_args(self, **kwargs):
        d = {}
        d["permc_spec"] = kwargs.get("permc_spec", None)
        d["diag_pivot_thresh"] = kwargs.get("diag_pivot_thresh", None)
        d["drop_tol"] = kwargs.get("drop_tol", None)
        d["relax"] = kwargs.get("relax", None)
        d["panel_size"] = kwargs.get("panel_size", None)
        return d

    def __extract_spilu_args(self, **kwargs):
        d = self.__extract_splu_args(**kwargs)
        d["drop_tol"] = kwargs.get("drop_tol", None)
        d["fill_factor"] = kwargs.get("fill_factor", None)
        d["drop_rule"] = kwargs.get("drop_rule", None)
        return d

    def __as_linear_operator(self, mat, sz=None):
        if sz is None:
            sz = mat.shape

        def mv(v):
            mat.solve(v)

        return spl.LinearOperator(sz, matvec=mv)
