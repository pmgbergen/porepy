"""The module contains the data class for forward mode automatic differentiation.
"""
from __future__ import annotations

from typing import Union
import numpy as np
import scipy.sparse as sps

AdType = Union[float, np.ndarray, sps.spmatrix, "Ad_array"]

__all__ = ["initAdArrays", "Ad_array"]


def initAdArrays(variables: list[np.ndarray]) -> list[Ad_array]:
    """Initialize a set of Ad_arrays.

    The variables' gradients will be taken with respect all variables jointly.

    Parameters:
        variables: A list of numpy arrays, each of which will be represented by an
            Ad_array.

    Returns:
        A list of Ad_arrays, each of which represents one of the variables in the
        ``variables`` list.

    """

    num_val_per_variable = [v.size for v in variables]
    ad_arrays: list[Ad_array] = []

    for i, val in enumerate(variables):
        # initiate zero jacobian
        n = num_val_per_variable[i]
        jac = [sps.csc_matrix((n, m)) for m in num_val_per_variable]
        # Set jacobian of variable i to I
        jac[i] = sps.diags(np.ones(num_val_per_variable[i])).tocsr()
        # initiate Ad_array
        jac = sps.bmat([jac])
        ad_arrays.append(Ad_array(val, jac))

    return ad_arrays


class Ad_array:
    """A class for representing differentiable quantities in a forward Ad mode.

    The class implements methods for arithmetic operations with floats, numpy arrays,
    scipy sparse matrices, and other ``Ad_arrays``. For these operations, the following
    general rules apply:
      * Scalars can be used for any arithmetic operation. As a convenience measure to
        limit the number of cases that mest be handled and maintained, the scalar must
        be a float.
      * Numpy arrays are assumed to be 1d and have the same size as the ``Ad_array``.
        Numpy arrays can be used for any operation except matrix multiplication (the @
        operator). *When adding, subtracting or multiplying a numpy array and an
        Ad_array, the Ad_array should be placed first, so, DO: Ad_array + numpy.array,
        DO NOT: numpy.array + Ad_array. The latter will give erratic behavior, see
        https://stackoverflow.com/a/6129099. Similarly, do not try to take
      * Scipy matrices can only be used for matrix-vector products (the @ operator), and
        then only for left multiplication. While right multiplication could technically
        work, depending on the size of the matrix, this is not the way the Ad framework
        is intended to be used, and so this operation is not supported.
      * Other Ad_arrays can be used with all arithmetic operations except the @
        operator.

    A violation of these rules will result in a ``ValueError``.

    Attributes:
        val (np.ndarray): The value of the Ad_array, stored as a 1d numpy array. jac
        (sps.spmatrix): The Jacobian matrix of the Ad_array, stored as a sparse
            matrix.

    """

    def __init__(self, val: np.ndarray, jac: sps.spmatrix) -> None:

        # Consistency checks, to limit the possibilities for errors when combining this
        # array with other objects.
        if val.ndim != 1:
            raise ValueError("The Ad array value should be one dimensional")
        if jac.shape[0] != val.size:
            raise ValueError(
                "The Jacobian matrix should have one row per array degree of freedom"
            )
        if jac.shape[1] < val.size:
            raise ValueError(
                """The Jacobian matrix should at least contain derivatives with respect
                to the variable itself"""
            )

        # Enforce float format of all data to limit the number of cases we need to
        # handle and test.
        self.val: np.ndarray = val.astype(float)
        """The value of the Ad_array, stored as a 1d numpy array."""

        self.jac: sps.spmatrix = jac.astype(float)
        """The Jacobian matrix of the Ad_array, stored as a sparse matrix."""

    def __repr__(self) -> str:
        s = f"Ad array of size {self.val.size}\n"
        s += f"Jacobian is of size {self.jac.shape} and has {self.jac.data.size}"
        s += " elements"
        return s

    def __add__(self, other: AdType) -> Ad_array:
        """Add the Ad_array to another object.

        Parameters:
            other: An object to be added to this object. See class documentation for
                restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An Ad_array which combines ``self`` and ``other``.

        """
        # Use class patterns for identfying the right case, see
        # https://stackoverflow.com/questions/67524641/convert-multiple-isinstance-checks-to-structural-pattern-matching
        # and https://peps.python.org/pep-0634/#class-patterns
        match other:
            case float():
                return Ad_array(self.val + other, self.jac)

            case np.ndarray():
                if other.ndim != 1:
                    raise ValueError("Only 1d numpy arrays can be added to Ad_arrays")
                return Ad_array(self.val + other, self.jac)

            case sps.spmatrix():
                raise ValueError("Sparse matrices cannot be added to Ad_arrays")

            case Ad_array():
                if self.val.size != other.val.size or self.jac.shape != other.jac.shape:
                    raise ValueError("Incompatible sizes for Ad_array addition")
                return Ad_array(self.val + other.val, self.jac + other.jac)

            case int():
                # Explicitly catch this case, since it will likely happen occasionally.
                # This can be circumvented by converting the int to a float.
                raise ValueError(
                    """Scalars should be converted to floats before parsing
                         in the Ad framework"""
                )

            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __radd__(self, other: AdType) -> Ad_array:
        """Add the Ad_array to another object.

        Parameters:
            other: An object to be added to this object. See class documentation for
                restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An Ad_array which combines ``self`` and ``other``.

        """
        return self.__add__(other)

    def __sub__(self, other: AdType) -> Ad_array:
        """Subtract another object from this Ad_array.

        Parameters:
            other: An object to be added to this object. See class documentation for
                restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An Ad_array which combines ``self`` and ``other``.

        """
        return self.__add__(-other)

    def __rsub__(self, other: AdType) -> Ad_array:
        """Subtract this Ad_array from another object.

        Parameters:
            other: An object to be added to this object. See class documentation for
                restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An Ad_array which subtracts ``self`` from ``other``.

        """
        # Calculate self - other and negative the answer (note the minus sign in front).
        return -self.__sub__(other)

    # TODO: EK removed these methods temporarily, since it is not clear if they are
    # used, and what the natural interpretation of the methods would be. Revisit this at
    # a later point.
    #
    # def __lt__(self, other): return self.val < _cast(other).val

    # def __le__(self, other):
    #    return self.val <= _cast(other).val

    # def __gt__(self, other):
    #    return self.val > _cast(other).val

    # def __ge__(self, other):
    #    return self.val >= _cast(other).val

    # def __eq__(self, other):
    #    return self.val == _cast(other).val

    def __mul__(self, other: AdType) -> Ad_array:
        """Elementwise product between two objects.

        Parameters:
            other: An object to be multiplied with this object. See class documentation
                for restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An Ad_array which multiplies ``self`` and ``other`` elementwise.

        """

        match other:
            case float():
                return Ad_array(self.val * other, self.jac * other)

            case np.ndarray():
                if other.ndim != 1:
                    raise ValueError("Only 1d numpy arrays can be added to Ad_arrays")
                # The below line will invoke numpy's __mul__ method on the values.
                new_val = self.val * other
                # The Jacobian will have its columns scaled with the values in other.
                # Achieve this by left-multiplying with other, represented as a diagonal
                # matrix.
                new_jac = self._diagvec_mul_jac(other)
                return Ad_array(new_val, new_jac)

            case sps.spmatrix():
                raise ValueError(
                    """Sparse matrices cannot be multiplied Ad_arrays elementwise.
                    Did you mean to use the @ operator?
                    """
                )

            case Ad_array():
                if self.val.size != other.val.size or self.jac.shape != other.jac.shape:
                    raise ValueError("Incompatible sizes for Ad_array addition")

                # For the values, use elementwise multiplication, as implemented by
                # numpy's __mul__ method
                new_val = self.val * other.val
                # Compute the derivative of the product using the product rule. Since
                # the gradients in jac is stored row-wise, the columns in self.jac
                # should be scaled with the values of other and vice versa.
                new_jac = self._diagvec_mul_jac(other.val) + other._diagvec_mul_jac(
                    self.val
                )
                return Ad_array(new_val, new_jac)

            case int():
                # This can be circumvented by converting the int to a float.
                raise ValueError(
                    """Scalars should be converted to floats before parsing
                         in the Ad framework"""
                )
            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __rmul__(self, other: AdType) -> Ad_array:
        """Elementwise product between two objects.

        Parameters:
            other: An object to be multiplied with this object. See class documentation
                for restrictions on admissible types for this function.

        Returns:
            An Ad_array which multiplies ``self`` and ``other`` elementwise.

        Raises:
            ValueError: If this represents an impermissible operation.

        """

        match other:
            case float() | sps.spmatrix() | np.ndarray() | int():
                # In these cases, there is no difference between left and right
                # multiplication, so we simply invoke the standard __mul__ function.
                return self.__mul__(other)

            case Ad_array():
                # The only way we can end up here is if other.__mul__(self) returns
                # NotImplemented, which makes no sense. Raise an error; if we ever end
                # up here, something is really wrong.
                raise RuntimeError("Something went wrong when multiplying to Ad_arrays")
            case _:
                raise ValueError(
                    f"Unknown type {type(other)} for Ad_array multiplication"
                )

    def __pow__(self, other: AdType) -> Ad_array:
        """Raise this Ad_array to the power of another object.

        Parameters:
            other: An object with exponent to which this Ad_array is raised. The power
                is implemented elementwise. See class documentation for restrictions on
                admissible types for this function.

        Returns:
            An Ad_array which represent ``other`` ** ``self`` elementwise.

        """

        match other:
            case float():
                # This is a polynomial, use standard rules for differentiation.
                new_val = self.val**other
                # Left-multiply jac with a diagonal-matrix version of the differentiated
                # polynomial, this will give the desired column-wise scaling of the
                # gradients.
                new_jac = self._diagvec_mul_jac(other * self.val ** (other - 1))
                return Ad_array(new_val, new_jac)

            case np.ndarray():
                if other.ndim != 1:
                    raise ValueError("Only 1d numpy arrays can be added to Ad_arrays")
                # This is a polynomial, but with different coefficients for each element
                # in self.val. Numpy can be picky on raising arrays to negative powers,
                # without EK ever understanding why, so we convert to a float
                # beforehand, just to be sure.
                new_val = self.val ** other.astype(float)
                # The Jacobian will have its columns scaled with the values in other,
                # again in array-form. Achieve this by left-multiplying with other,
                # represented as a diagonal matrix.
                new_jac = self._diagvec_mul_jac(other * (self.val ** (other - 1)))
                return Ad_array(new_val, new_jac)

            case sps.spmatrix():
                raise ValueError("Cannot raise Ad_arrays to power of sparse matrices")

            case Ad_array():
                if self.val.size != other.val.size or self.jac.shape != other.jac.shape:
                    raise ValueError("Incompatible sizes for Ad_array addition")

                # This is an expression of the type f = x^y, with derivative
                #
                #   df = (y * x ** (y-1)) * dx + x^y * log(x) * dy
                #
                # Compute the value using numpy's power method. Convert to float to
                # avoid spurious behavior form numpy, just to be sure.
                new_val = self.val ** other.val.astype(float)
                # The derivative, computed by the chain rule.
                new_jac = self._diagvec_mul_jac(
                    other.val * self.val ** (other.val.astype(float) - 1.0)
                ) + other._diagvec_mul_jac(
                    self.val ** other.val.astype(float) * np.log(self.val)
                )

                return Ad_array(new_val, new_jac)

            case int():
                # This can be circumvented by converting the int to a float.
                raise ValueError(
                    """Scalars should be converted to floats before parsing
                         in the Ad framework"""
                )
            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __rpow__(self, other: AdType) -> Ad_array:
        """Raise another object to the power of this Ad_array.

        Parameters:
            other: An object which should be raised to the power of this Ad_array.
                The power is implemented elementwise. See class documentation for
                restrictions on admissible types for this function.

        Returns:
            An Ad_array which represent ``other`` ** ``self`` elementwise.

        """

        match other:
            case float():
                # This is an exponent of type number ** x
                new_val = other**self.val
                # Left-multiply jac with a diagonal-matrix version of the differentiated
                # polynomial, this will give the desired column-wise scaling of the
                # gradients.
                new_jac = self._diagvec_mul_jac((other**self.val) * np.log(other))
                return Ad_array(new_val, new_jac)

            case np.ndarray():
                if other.ndim != 1:
                    raise ValueError("Only 1d numpy arrays can be added to Ad_arrays")
                # This is an exponent with different coefficients for each element
                # in self.val. Numpy can be picky on raising arrays to negative powers,
                # without EK ever understanding why, so we convert to a float
                # beforehand, just to be sure.
                new_val = other.astype(float) ** self.val
                # The Jacobian will have its columns scaled with the values in other,
                # again in array-form. Achieve this by left-multiplying with other,
                # represented as a diagonal matrix.
                new_jac = self._diagvec_mul_jac((other**self.val) ** np.log(self.val))
                return Ad_array(new_val, new_jac)

            case sps.spmatrix():
                raise ValueError(
                    "Cannot raise sparse matrices to the power of Ad arrays"
                )

            case Ad_array():
                if self.val.size != other.val.size or self.jac.shape != other.jac.shape:
                    raise ValueError("Incompatible sizes for Ad_array addition")

                return other.__pow__(self)

            case int():
                # This can be circumvented by converting the int to a float.
                raise ValueError(
                    """Scalars should be converted to floats before parsing
                         in the Ad framework"""
                )
            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __truediv__(self, other: AdType) -> Ad_array:
        """Divide this Ad_array by another object.

        Parameters:
            other: An object which should divide this Ad_array. The division is
                implemented elementwise. See class documentation for restrictions on
                admissible types for this function.

        Returns:
            An Ad_array which represent ``self`` / ``other`` elementwise.

        """

        match other:
            case float():
                # Division by float is straightforward, elementwise.
                new_val = self.val / other
                new_jac = self.jac / other
                return Ad_array(new_val, new_jac)

            case np.ndarray():
                if other.ndim != 1:
                    raise ValueError("Only 1d numpy arrays can be added to Ad_arrays")
                # This is an exponent with different coefficients for each element
                # in self.val. Numpy can be picky on raising arrays to negative powers,
                # without EK ever understanding why, so we convert to a float
                # beforehand, just to be sure.
                new_val = self.val * other.astype(float) ** (-1.0)
                # The Jacobian will have its columns scaled with the values in other,
                # again in array-form. Achieve this by left-multiplying with other,
                # represented as a diagonal matrix.
                new_jac = self._diagvec_mul_jac(other.astype(float) ** (-1.0))
                return Ad_array(new_val, new_jac)

            case sps.spmatrix():
                raise ValueError(
                    "Cannot raise sparse matrices to the power of Ad arrays"
                )

            case Ad_array():
                if self.val.size != other.val.size or self.jac.shape != other.jac.shape:
                    raise ValueError("Incompatible sizes for Ad_array addition")

                return self.__mul__(other.__pow__(-1.0))

            case int():
                # This can be circumvented by converting the int to a float.
                raise ValueError(
                    """Scalars should be converted to floats before parsing
                         in the Ad framework"""
                )
            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __rtruediv__(self, other: AdType) -> Ad_array:
        """Divide another object by this Ad_array.

        Parameters:
            other: An object which should be divided by this Ad_array. The division is
                implemented elementwise. See class documentation for restrictions on
                admissible types for this function.

        Returns:
            An Ad_array which represent ``other`` / ``self`` elementwise.

        """

        match other:
            case float() | np.ndarray() | sps.spmatrix() | int():
                # Divide a float or a numpy array by self is the same as raising self to
                # the power of -1 and multiplying by the float. The multiplication will
                # end upcalling self.__mul__, which will do the right checks for numpy
                # arrays and sparse matrices.
                return other * self.__pow__(-1.0)

            case Ad_array():
                if self.val.size != other.val.size or self.jac.shape != other.jac.shape:
                    raise ValueError("Incompatible sizes for Ad_array addition")

                return other.__mul__(self.__pow__(-1.0))

            case int():
                # This can be circumvented by converting the int to a float.
                raise ValueError(
                    """Scalars should be converted to floats before parsing
                         in the Ad framework"""
                )
            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __matmul__(self, other: AdType) -> Ad_array:
        """Do a matrix multiplication between this Ad_array and another object.

        Parameters:
            other: An object which should be right multiplied with this Ad_array. See
                class documentation for restrictions on admissible types for this
                function.

        Returns:
            An Ad_array which represent ``self`` @ ``other`` elementwise.

        """

        match other:
            case float():
                return self.__mul__(other)

            case np.ndarray() | Ad_array() | int():
                raise ValueError(
                    """Cannot perform matrix multiplication between an Ad_array and a"""
                    f""" {type(other)}"""
                )

            case sps.spmatrix():
                # This goes against the way equations should be formulated in the AD
                # framework, variables should not be right-multiplied by anything. Raise
                # a value error to make sure this is not done.
                raise ValueError(
                    """Ad_arrays should only be left-multiplied by sparse matrices."""
                )

            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __rmatmul__(self, other):
        """Do a matrix multiplication between another object and this Ad_array.

        Parameters:
            other: An object which should be left multiplied with this Ad_array. See
                class documentation for restrictions on admissible types for this
                function.

        Returns:
            An Ad_array which represent ``other`` @ ``self`` elementwise.

        """
        match other:
            case float():
                return self.__mul__(other)

            case np.ndarray() | Ad_array() | int():
                raise ValueError(
                    """Cannot perform matrix multiplication between an Ad_array and a"""
                    f""" {type(other)}"""
                )

            case sps.spmatrix():
                # This is the standard matrix-vector multiplication
                if self.jac.shape[0] != other.shape[1]:
                    raise ValueError(
                        """Dimension mismatch between sparse matrix and Ad_array"""
                    )
                new_val = other @ self.val
                new_jac = other @ self.jac
                return Ad_array(new_val, new_jac)

            case _:
                raise ValueError(f"Unknown type {type(other)} for Ad_array addition")

    def __neg__(self):
        b = self.copy()
        b.val = -b.val
        b.jac = -b.jac
        return b

    def copy(self) -> Ad_array:
        """Return a copy of this Ad_array.

        Returns:
            A deep copy of this Ad_array.

        """
        b = Ad_array(self.val.copy(), self.jac.copy())
        return b

    def _diagvec_mul_jac(self, a: np.ndarray) -> sps.spmatrix:
        A = sps.diags(a)

        return A * self.jac

    def _jac_mul_diagvec(self, a: np.ndarray) -> sps.spmatrix:
        A = sps.diags(a)

        return self.jac * A


# EK: This can likely go together with the __le__ methods etc., but do not delete them
# just yet.
# def _cast(variables):
#    if isinstance(variables, list):
#        out_var = []
#        for var in variables:
#            if isinstance(var, Ad_array):
#                out_var.append(var)
#            else:
#                out_var.append(Ad_array(var))
#    else:
#        if isinstance(variables, Ad_array):
#            out_var = variables
#        else:
#            out_var = Ad_array(variables)
#    return out_var
