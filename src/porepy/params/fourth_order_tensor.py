import numpy as np


class FourthOrderTensor(object):
    """ Cell-wise representation of fourth order tensor.

    For each cell, there are dim^4 degrees of freedom, stored in a
    dim^2 * dim^2 matrix (exactly how to convert between 2D and 4D matrix
    is not clear to me a the moment, but in pratcise there is sufficient
    symmetry in the tensors for the question to be irrelevant).

    The tensor is thus stored in a (dim^2,dim^2,nc) matrix, named c.

    The only constructor available for the moment is based on the lame parameters,
    e.g. using two degrees of freedom. A third parameter phi is also present,
    but this has never been used.

    Primary usage for the class is for mpsa discretizations. Other applications
    have not been tested.

    Attributes:
        c - numpy.ndarray, dimensions (dim^2,dim^2,nc), cell-wise representation of

    """
    def __init__(self, dim, mu, lmbda, phi=None):
        """ Constructor for fourth order tensor on Lame-parameter form

        Parameters
        ----------
        dim (int) dimension, should be 2 or 3
        mu (numpy.ndarray), First lame parameter, 1-D, one value per cell
        lmbda (numpy.ndarray), Second lame parameter, 1-D, one value per cell
        phi (Optional numpy.ndarray), 1-D one value per cell, never been used.

        """

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
            phi = 0 * mu  # Default value for phi is zero
        elif not isinstance(phi, np.ndarray):
            raise ValueError("Phi should be a numpy array")
        elif not phi.ndim == 1:
            raise ValueError("Phi should be 1-D")
        elif phi.size != lmbda.size:
            raise ValueError("Phi and Lmbda should have the same length")

        # Basis for the contributions of mu, lmbda and phi is hard-coded
        if dim == 2:
            mu_mat = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 2, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1]])

            lmbda_mat = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1]])

            phi_mat = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1, 0]])


        else:  # Dimension is either 2 or 3
            mu_mat = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 2, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 1, 0],
                               [0, 0, 1, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 2]])
            lmbda_mat = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1]])
            phi_mat = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1, 0]])

        # Expand dimensions to prepare for cell-wise representation
        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]
        phi_mat = phi_mat[:, :, np.newaxis]

        c = mu_mat * mu + lmbda_mat * lmbda + phi_mat * phi
        self.c = c

    def copy(self):
        return np.copy(self.c)
