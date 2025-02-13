"""
The tensor module contains classes for second and fourth order tensors, intended e.g.
for representation of permeability and stiffness, respectively.
"""

from __future__ import annotations

from typing import Optional, cast

import numpy as np


class Tensor:
    """Class of the common parts of the second and fourth order tensors."""

    values: np.ndarray

    @staticmethod
    def constitutive_parameters(self) -> list[str]:
        """Strings for the constitutive parameters found in the tensor.

        Returns:
            A list of the constitutive parameter names.

        """
        # Each constitutive parameter is assumed to be an array of shape (nc,)
        return []

    def restrict_to_cells(self, cells: np.ndarray) -> None:
        """Restrict constitutive parameters to cells.

        Used in the case of problems being split into several sub-problems.

        Parameters:
            cells: Indices of the cells which the constitutive parameters should be
                restricted to.

        """
        # Restrict all constitutive parameters.
        for field in self.constitutive_parameters(self):
            vals = cast(np.ndarray, getattr(self, field))
            setattr(self, field, vals[cells])

        # Restrict the values representation of the tensor.
        self.values = self.values[::, ::, cells]


class SecondOrderTensor(Tensor):
    """Cell-wise permeability represented.

    The permeability is always 3-dimensional (since the geometry is always 3D), however,
    1D and 2D problems are accommodated by assigning unit values to kzz and kyy, and no
    cross terms.
    """

    def __init__(
        self,
        kxx: np.ndarray,
        kyy: Optional[np.ndarray] = None,
        kzz: Optional[np.ndarray] = None,
        kxy: Optional[np.ndarray] = None,
        kxz: Optional[np.ndarray] = None,
        kyz: Optional[np.ndarray] = None,
    ):
        """Initialize permeability

        Parameters:
            kxx: Nc array, with cell-wise values of kxx permeability.
            kyy: Nc array of kyy. Default equal to kxx.
            kzz: Nc array of kzz. Default equal to kxx. Not used if dim < 3.
            kxy: Nc array of kxy. Defaults to zero.
            kxz: Nc array of kxz. Defaults to zero. Not used if dim < 3.
            kyz: Nc array of kyz. Defaults to zero. Not used if dim < 3.

        Raises:
            ValueError if the permeability is not positive definite.

        """
        Nc = kxx.size
        perm = np.zeros((3, 3, Nc))

        if np.any(kxx < 0):
            raise ValueError(
                "Tensor is not positive definite because of components in x-direction"
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
                "Tensor is not positive definite because of components in y-direction"
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
                "Tensor is not positive definite because of components in z-direction"
            )

        perm[2, 0, ::] = kxz
        perm[0, 2, ::] = kxz
        perm[2, 1, ::] = kyz
        perm[1, 2, ::] = kyz
        perm[2, 2, ::] = kzz

        self.values = perm

    def copy(self) -> SecondOrderTensor:
        """Define a deep copy of the tensor.

        Returns:
            SecondOrderTensor: New tensor with identical fields, but separate arrays (in
                the memory sense).
        """

        kxx = self.values[0, 0].copy()
        kxy = self.values[1, 0].copy()
        kyy = self.values[1, 1].copy()
        kxz = self.values[2, 0].copy()
        kyz = self.values[2, 1].copy()
        kzz = self.values[2, 2].copy()

        return SecondOrderTensor(kxx, kxy=kxy, kxz=kxz, kyy=kyy, kyz=kyz, kzz=kzz)

    def rotate(self, R: np.ndarray) -> None:
        """Rotate the permeability given a rotation matrix.

        Parameter:
            R: a rotation matrix 3x3.

        """
        self.values = np.tensordot(R.T, np.tensordot(R, self.values, (1, 0)), (0, 1))

    def __str__(self) -> str:
        # While the tensor is always initiated as 3x3 on each cell, it may be restricted
        # to 2x2, e.g., to comply with assumptions underlying the mpfa implementation.
        # Thus we report on the tensor shape, in addition ot the number of cells.
        s = f"Second order tensor of shape {self.values.shape[:2]}"
        s += f" defined on {self.values.shape[2]} cells"
        return s

    def __repr__(self) -> str:
        return self.__str__()


class FourthOrderTensor(Tensor):
    """Cell-wise representation of fourth order tensor.

    For each cell, there are dim^4 degrees of freedom, stored in a 3^2 * 3^2 matrix
    (exactly how to convert between 2D and 4D matrix is not clear to me a the moment,
    but in practise there is sufficient symmetry in the tensors for the question to be
    irrelevant).

    The only constructor available for the moment is based on the Lame parameters, e.g.
    using two degrees of freedom. A third parameter phi is also present, but this has
    never been used.

    Primary usage for the class is for mpsa discretizations. Other applications have not
    been tested.

    """

    def __init__(
        self, mu: np.ndarray, lmbda: np.ndarray, phi: Optional[np.ndarray] = None
    ):
        """Constructor for fourth order tensor on Lame-parameter form.

        Raises:
            ValueError if mu, lmbda or phi (if not None) are not 1d arrays.
            ValueError if the lengths of mu, lmbda and phi (if not None) are not
                matching.
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
        """Nc array of first Lamé parameter."""
        self.mu = mu
        """Nc array of shear modulus (second Lamé parameter)."""
        self.phi = phi
        """Nc array of ... spør Jan eller Eirik."""

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

        # Expand dimensions to prepare for cell-wise representation.
        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]
        phi_mat = phi_mat[:, :, np.newaxis]

        c = mu_mat * mu + lmbda_mat * lmbda + phi_mat * phi
        self.values = c
        """Values of the stiffness tensor as a (3^2, 3^2, Nc) array."""

    def copy(self) -> FourthOrderTensor:
        """Define a deep copy of the tensor.

        Returns:
            FourthOrderTensor: New tensor with identical fields, but separate arrays (in
                the memory sense).

        """
        C = FourthOrderTensor(mu=self.mu.copy(), lmbda=self.lmbda.copy())
        C.values = self.values.copy()
        return C

    @staticmethod
    def constitutive_parameters(self) -> list:
        """Strings for the constitutive parameters found in the fourth order tensor.

        Returns:
            A list of the constitutive parameter names.

        """
        return [
            "mu",
            "lmbda",
            # Legge inn phi som attributt??
        ]

    def __str__(self) -> str:
        s = f"Fourth order tensor defined on {self.values.shape[2]} cells.\n"

        # During discretization (e.g., mpsa), the original 3x3x3x3 may be restricted to
        # 2d, thus (2, 2, 2, 2), to comply with assumptions in the implementation of the
        # discretization. Therefore, we report on the shape of the tensor. The notation
        # below acknowledges the full fourth-order nature of this tensor (as opposed to
        # the storage format in self.values, which joins two and two dimensions).
        if self.values.shape[:2] == (4, 4):
            s += "Each cell has a tensor of shape (2, 2, 2, 2)"
        elif self.values.shape[:2] == (9, 9):
            s += "Each cell has a tensor of shape (3, 3, 3, 3)"
        else:
            # In EK's understanding, this should never happen. Give a fair description
            # of the situation, and hope the user knows what is going on.
            s += f"The tensor has shape {self.values[:, :, 0].shape}."
            s += "It is unclear how this came about."

        return s

    def __repr__(self) -> str:
        return self.__str__()
