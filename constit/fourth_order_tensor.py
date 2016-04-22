import numpy as np


class FourthOrderTensor(object):

    def __init__(self, dim, mu, lmbda, phi=None):

        # Check arguments
        if dim > 3 or dim < 2:
            raise ValueError('Dimension should be between 1 and 3')

        if not isinstance(mu, np.ndarray):
            raise ValueError("Input mu should be a numpy array")
        if not isinstance(lmbda, np.ndarray):
            raise ValueError("Input lmbda should be a numpy array")
        if not mu.ndim == 1:
            raise ValueError("mu should be 1-D")
        if not lmbda.ndim == 1:
            raise ValueError("Lmbda should be 1-D")
        if mu.size != lmbda.size:
            raise ValueError("Mu and lmbda should have the same length")

        if phi is None:
            phi = 0 * mu
        elif not isinstance(phi, np.ndarray):
            raise ValueError("Phi should be a numpy array")
        elif not phi.ndim == 1:
            raise ValueError("Phi should be 1-D")
        elif phi.size != lmbda.size:
            raise ValueError("Phi and Lmbda should have the same length")

        nc = mu.size
        d2 = dim * dim
        stiffness = np.zeros((d2, d2, nc))

        if dim == 2:
            mu_mat = np.array([[2, 0, 0, 0],
                               [0, 1, 1, 0],
                               [0, 1, 1, 0],
                               [0, 0, 0, 2]])
            lmbda_mat = np.array([[1, 0, 0, 1],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [1, 0, 0, 1]])
            phi_mat = np.array([[0, 1, 1, 0],
                               [1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [0, 1, 1, 0]])

        elif dim == 3:
            mu_mat = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 2, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 1, 0],
                               [0, 0, 1, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 2]
                               ])
            lmbda_mat = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1]
                                  ])
            phi_mat = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1, 0]
                                ])

        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]
        phi_mat = phi_mat[:, :, np.newaxis]

        c = mu_mat * mu + lmbda_mat * lmbda + phi_mat * phi
        self.c = c
