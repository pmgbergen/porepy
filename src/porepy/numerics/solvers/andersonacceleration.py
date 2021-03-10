import numpy as np
import scipy as sp


class AndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(self, dimension, depth):

        self._dimension = dimension
        self._depth = depth

        # Initialize arrays for iterates.
        self.reset()
        self._fkm1: np.ndarray = self._Fk.copy()
        self._gkm1: np.ndarray = self._Gk.copy()

    def reset(self):
        self._Fk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in increments
        self._Gk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in fixed point applications

    def apply(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:

        # gk : application of some fixed point iteration onto approximation xk, i.e., g(xk).
        # fk : residual g(xk) - xk; in general some increment
        # iteration : current iteration count

        if iteration == 0:
            self._Fk = np.zeros((self._dimension, self._depth))  # changes in increments
            self._Gk = np.zeros(
                (self._dimension, self._depth)
            )  # changes in fixed point applications

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk - self._fkm1
            self._Gk[:, col] = gk - self._gkm1

            # Solve least squares problem
            lstsq_solution = sp.linalg.lstsq(self._Fk[:, 0:mk], fk)
            gamma_k = lstsq_solution[0]
            # Do the mixing
            xkp1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
        else:
            xkp1 = gk

        # Store values for next iteration
        self._fkm1 = fk.copy()
        self._gkm1 = gk.copy()

        return xkp1
