import numpy as np
from scipy.linalg import lstsq


class AndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(
        self,
        dimension: int,
        depth: int,
        constrain_acceleration: bool = False,
        regularization_parameter: float = 0.0,
    ) -> None:
        self._dimension = int(dimension)
        self._depth = int(depth)
        self._constrain_acceleration: bool = bool(constrain_acceleration)
        self._reg_param: float = float(regularization_parameter)

        # Initialize arrays for iterates.
        self.reset()
        self._fkm1: np.ndarray = np.zeros(self._dimension)
        self._gkm1: np.ndarray = np.zeros(self._dimension)

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
            self.reset()

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes.
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk - self._fkm1
            self._Gk[:, col] = gk - self._gkm1

            # Solve least squares problem.
            A = self._Fk[:, 0:mk]
            b = fk
            if self._constrain_acceleration:
                A = np.vstack((A, np.ones((1, self._depth))))
                b = np.concatenate((b, np.ones(1)))

            direct_solve = False

            if self._reg_param > 0:
                b = A.T @ b
                A = A.T @ A + self._reg_param * np.eye(A.shape[1])
                direct_solve = np.linalg.matrix_rank(A) >= A.shape[1]

            if direct_solve:
                gamma_k = np.linalg.solve(A, b)
            else:
                gamma_k = lstsq(A, b)[0]

            # Do the mixing
            x_k_plus_1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
        else:
            x_k_plus_1 = gk

        # Store values for next iteration.
        self._fkm1 = fk.copy()
        self._gkm1 = gk.copy()

        return x_k_plus_1
