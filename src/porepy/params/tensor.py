# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:22:25 2016

@author: keile

The tensor module contains classes for second and fourth order tensors,
intended e.g. for representation of permeability and stiffness, respectively.
"""
import numpy as np

import porepy as pp

module_sections = ["parameters"]


class SecondOrderTensor(object):
    """Cell-wise permeability represented by (3 ,3 ,Nc)-matrix.

    The permeability is always 3-dimensional (since the geometry is always 3D),
    however, 1D and 2D problems are accomodated by assigning unit values to kzz
    and kyy, and no cross terms.
    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, kxx, kyy=None, kzz=None, kxy=None, kxz=None, kyz=None):
        """Initialize permeability

        Parameters:
            kxx (double): Nc array, with cell-wise values of kxx permeability.
            kyy (optional, double): Nc array of kyy. Default equal to kxx.
            kzz (optional, double): Nc array of kzz. Default equal to kxx.
                Not used if dim < 3.
            kxy (optional, double): Nc array of kxy. Defaults to zero.
            kxz (optional, double): Nc array of kxz. Defaults to zero.
                Not used if dim < 3.
            kyz (optional, double): Nc array of kyz. Defaults to zero.
                Not used if dim < 3.

        Raises:
            ValueError if the permeability is not positive definite.
        """
        Nc = kxx.size
        perm = np.zeros((3, 3, Nc))

        if np.any(kxx < 0):
            raise ValueError(
                "Tensor is not positive definite because of "
                "components in x-direction"
            )

        perm[0, 0, ::] = kxx

        # Assign unit values to the diagonal, these are overwritten later if
        # the specified data
        perm[1, 1, ::] = 1
        perm[2, 2, ::] = 1

        if kyy is None:
            kyy = kxx
        if kxy is None:
            kxy = 0 * kxx
        # Onsager's principle - tensor should be positive definite
        if np.any((kxx * kyy - kxy * kxy) < 0):
            raise ValueError(
                "Tensor is not positive definite because of "
                "components in y-direction"
            )

        perm[1, 0, ::] = kxy
        perm[0, 1, ::] = kxy
        perm[1, 1, ::] = kyy

        if kyy is None:
            kyy = kxx
        if kzz is None:
            kzz = kxx
        if kxy is None:
            kxy = 0 * kxx
        if kxz is None:
            kxz = 0 * kxx
        if kyz is None:
            kyz = 0 * kxx
        # Onsager's principle - tensor should be positive definite
        if np.any(
            (
                kxx * (kyy * kzz - kyz * kyz)
                - kxy * (kxy * kzz - kxz * kyz)
                + kxz * (kxy * kyz - kxz * kyy)
            )
            < 0
        ):
            raise ValueError(
                "Tensor is not positive definite because of "
                "components in z-direction"
            )

        perm[2, 0, ::] = kxz
        perm[0, 2, ::] = kxz
        perm[2, 1, ::] = kyz
        perm[1, 2, ::] = kyz
        perm[2, 2, ::] = kzz

        self.values = perm

    @pp.time_logger(sections=module_sections)
    def copy(self):
        """
        Define a deep copy of the tensor.

        Returns:
            SecondOrderTensor: New tensor with identical fields, but separate
                arrays (in the memory sense).
        """

        kxx = self.values[0, 0, :].copy()
        kxy = self.values[1, 0, :].copy()
        kyy = self.values[1, 1, :].copy()

        kxz = self.values[2, 0, :].copy()
        kyz = self.values[2, 1, :].copy()
        kzz = self.values[2, 2, :].copy()

        return SecondOrderTensor(kxx, kxy=kxy, kxz=kxz, kyy=kyy, kyz=kyz, kzz=kzz)

    @pp.time_logger(sections=module_sections)
    def rotate(self, R):
        """
        Rotate the permeability given a rotation matrix.

        Parameter:
            R: a rotation matrix 3x3
        """
        self.values = np.tensordot(R.T, np.tensordot(R, self.values, (1, 0)), (0, 1))


# ----------------------------------------------------------------------#


class FourthOrderTensor(object):
    """Cell-wise representation of fourth order tensor.

    For each cell, there are dim^4 degrees of freedom, stored in a
    3^2 * 3^2 matrix (exactly how to convert between 2D and 4D matrix
    is not clear to me a the moment, but in pratcise there is sufficient
    symmetry in the tensors for the question to be irrelevant).

    The only constructor available for the moment is based on the lame parameters,
    e.g. using two degrees of freedom. A third parameter phi is also present,
    but this has never been used.

    Primary usage for the class is for mpsa discretizations. Other applications
    have not been tested.

    Attributes:
        values - numpy.ndarray, dimensions (3^2, 3^2, nc), cell-wise
            representation of the stiffness matrix.
        lmbda (np.ndarray, size: num_cells): First Lame parameter
        mu (np.ndarray, size: num_cells): Second Lame parameter

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, mu, lmbda, phi=None):
        """Constructor for fourth order tensor on Lame-parameter form

        Parameters
        ----------
        mu (numpy.ndarray), First lame parameter, 1-D, one value per cell
        lmbda (numpy.ndarray), Second lame parameter, 1-D, one value per cell
        phi (Optional numpy.ndarray), 1-D one value per cell, never been used.

        """

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

        # Save lmbda and mu, can be useful to have in some cases
        self.lmbda = lmbda
        self.mu = mu

        # Basis for the contributions of mu, lmbda and phi is hard-coded
        mu_mat = np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2],
            ]
        )
        lmbda_mat = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
            ]
        )
        phi_mat = np.array(
            [
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
            ]
        )

        # Expand dimensions to prepare for cell-wise representation
        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]
        phi_mat = phi_mat[:, :, np.newaxis]

        c = mu_mat * mu + lmbda_mat * lmbda + phi_mat * phi
        self.values = c

    @pp.time_logger(sections=module_sections)
    def copy(self):
        C = FourthOrderTensor(mu=self.mu, lmbda=self.lmbda)
        C.values = self.values.copy()
        return C
