# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:22:25 2016

@author: keile
"""
import copy
import numpy as np


class SecondOrderTensor(object):
    """ Cell-wise permeability represented by (3 ,3 ,Nc)-matrix.

    The permeability is always 3-dimensional (since the geometry is always 3D),
    however, 1D and 2D problems are accomodated by assigning unit values to kzz
    and kyy, and no cross terms.
    """

    def __init__(self, dim, kxx, kyy=None, kzz=None,
                 kxy=None, kxz=None, kyz=None):
        """ Initialize permeability

        Parameters:
            dim (int): Dimension, should be between 1 and 3.
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
        if dim > 3 or dim < 0:
            raise ValueError('Dimension should be between 1 and 3')

        self.dim = dim

        Nc = kxx.size
        perm = np.zeros((3, 3, Nc))

        if not np.all(kxx > 0):
            raise ValueError('Tensor is not positive definite because of '
                             'components in x-direction')

        perm[0, 0, ::] = kxx

        # Assign unit values to the diagonal, these are overwritten later if
        # the specified dimension requires it.
        perm[1, 1, ::] = 1
        perm[2, 2, ::] = 1

        if dim > 1:
            if kyy is None:
                kyy = kxx
            if kxy is None:
                kxy = 0 * kxx
            # Onsager's principle
            if not np.all((kxx * kyy - kxy * kxy) > 0):
                raise ValueError('Tensor is not positive definite because of '
                                 'components in y-direction')

            perm[1, 0, ::] = kxy
            perm[0, 1, ::] = kxy
            perm[1, 1, ::] = kyy

        if dim > 2:
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
            # Onsager's principle
            if not np.all((kxx * (kyy * kzz - kyz * kyz) -
                           kxy * (kyz * kzz - kxz * kyz) +
                           kxz * (kxy * kyz - kxz * kyy)) > 0):
                raise ValueError('Tensor is not positive definite because of '
                                 'components in z-direction')

            perm[2, 0, ::] = kxz
            perm[0, 2, ::] = kxz
            perm[2, 1, ::] = kyz
            perm[1, 2, ::] = kyz
            perm[2, 2, ::] = kzz

        self.perm = perm

    def copy(self):
        """
        Define a deep copy of the tensor.

        Returns:
            SecondOrderTensor: New tensor with identical fields, but separate
                arrays (in the memory sense).
        """

        if self.dim == 2:

            kxx = self.perm[0, 0, :].copy()
            kxy = self.perm[1, 0, :].copy()
            kyy = self.perm[1, 1, :].copy()

            return SecondOrderTensor(self.dim, kxx, kxy=kxy, kyy=kyy)
        else:
            kxx = self.perm[0, 0, :].copy()
            kxy = self.perm[1, 0, :].copy()
            kyy = self.perm[1, 1, :].copy()

            kxz = self.perm[2, 0, :].copy()
            kyz = self.perm[2, 1, :].copy()
            kzz = self.perm[2, 2, :].copy()

            return SecondOrderTensor(self.dim, kxx, kxy=kxy, kxz=kxz, kyy=kyy,
                                     kyz=kyz, kzz=kzz)
