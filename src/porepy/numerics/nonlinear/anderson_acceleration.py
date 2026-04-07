import numpy as np
from scipy.linalg import lstsq


class AndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(self, dimension, depth) -> None:
        self._dimension = dimension
        self._depth = depth

        # Initialize arrays for iterates.
        self.reset()
        self._fkm1: np.ndarray = self._Fk.copy()
        self._gkm1: np.ndarray = self._Gk.copy()

    def reset(self) -> None:
        self._Fk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in increments
        self._Gk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in fixed point applications

    def apply(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:
        """Apply Anderson acceleration.

        Parameters:
            gk: application of some fixed point iteration onto approximation xk, i.e.,
                g(xk).
            fk: residual g(xk) - xk; in general some increment.
            iteration: current iteration count.

        Returns:
            Modified application of fixed point approximation after acceleration, i.e.,
            the new iterate xk+1.

        """

        if iteration == 0:
            self._Fk = np.zeros((self._dimension, self._depth))  # changes in increments
            self._Gk = np.zeros(
                (self._dimension, self._depth)
            )  # changes in fixed point applications

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes.
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk - self._fkm1
            self._Gk[:, col] = gk - self._gkm1

            # Solve least squares problem.
            lstsq_solution = lstsq(self._Fk[:, 0:mk], fk)
            gamma_k = lstsq_solution[0]
            # Do the mixing
            x_k_plus_1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
        else:
            x_k_plus_1 = gk

        # Store values for next iteration.
        self._fkm1 = fk.copy()
        self._gkm1 = gk.copy()

        return x_k_plus_1


class AndersonAccelerationFiltering:
    """Anderson acceleration Algorithm 4 as described by An and Jia and Walker in doi.org/10.1016/j.jcp.2017.06.031.
    """

    def __init__(self, dimension, depth, filtering: bool = True, drop_tol: float = 1e10) -> None:
        self._dimension = dimension
        self._depth = depth
        self._filtering = filtering
        self._drop_tol = drop_tol # Tolerance for condition number of Fk
        
        # Initialize arrays for iterates.
        self.reset()
        self._fkm1: np.ndarray = self._Fk.copy()
        self._gkm1: np.ndarray = self._Gk.copy()

    def reset(self) -> None:
        self._Fk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in increments
        self._Gk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in fixed point applications
        self._mk = 0 # Tracks current depth of the history

    def apply(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:
        """Apply Anderson acceleration.

        Parameters:
            gk: application of some fixed point iteration onto approximation xk, i.e.,
                g(xk).
            fk: residual g(xk) - xk; in general some increment.
            iteration: current iteration count.

        Returns:
            Modified application of fixed point approximation after acceleration, i.e.,
            the new iterate xk+1.

        """

        if iteration == 0:
            self.reset()
            x_k_plus_1 = gk
        else:
            if self._mk < self._depth:
                col = self._mk
                self._mk += 1
            else:
                self._Fk[:,:-1] = self._Fk[:, 1:]
                self._Gk[:,:-1] = self._Gk[:, 1:]
                col = self._depth-1
            
            # Build matrices of changes.
            self._Fk[:, col] = fk - self._fkm1
            self._Gk[:, col] = gk - self._gkm1
            
            # Drop the oldest columns if the condition number is too large.
            if self._filtering:
                while self._mk > 1 and np.linalg.cond(self._Fk[:, :self._mk]) > self._drop_tol:
                    # Drop the oldest column (index 0) and shift the rest left
                    self._Fk[:, :self._mk - 1] = self._Fk[:, 1:self._mk]
                    self._Gk[:, :self._mk - 1] = self._Gk[:, 1:self._mk]
                    self._mk -= 1
                    
            # Solve least squares problem.
            lstsq_solution = lstsq(self._Fk[:, 0:self._mk], fk)
            gamma_k = lstsq_solution[0]
            # Do the mixing
            x_k_plus_1 = gk - np.dot(self._Gk[:, 0:self._mk], gamma_k)

        # Store values for next iteration.
        self._fkm1 = fk.copy()
        self._gkm1 = gk.copy()

        return x_k_plus_1

