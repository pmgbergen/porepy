"""The module contains the data class for forward mode automatic differentiation."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

__all__ = [
    "AdArrayBase",
    "AdArray",
    "initAdArrays",
    "DiagonalAdArray",
    "initialize_diagonal_ad_arrays",
    "initialize_partial_ad_array",
]

AdType = Union[int, float, np.ndarray, sps.spmatrix, sps.sparray, "AdArrayBase"]

_SPARSE_TYPES = (sps.spmatrix, sps.sparray)
"""Convenience tuple for isinstance checks accepting both scipy sparse matrix and
sparse array types."""


def _check_1d(val: np.ndarray) -> None:
    """Raise a ValueError unless ``val`` is a one-dimensional array."""
    if val.ndim != 1:
        raise ValueError("The Ad array value should be one-dimensional")


def _as_float(
    array: np.ndarray | sps.spmatrix | sps.sparray,
) -> np.ndarray | sps.spmatrix | sps.sparray:
    """Coerce a dense array, or the data of a sparse matrix/array, to float dtype.

    Enforcing float format for all data limits the number of cases that need to be
    handled and tested elsewhere.

    """
    if isinstance(array, _SPARSE_TYPES):
        if array.data.dtype != float:
            # astype returns a new matrix, so the caller's matrix is left alone.
            return array.astype(float)
        return array
    if array.dtype != float:
        array = array.astype(float)
    return array


def initialize_partial_ad_array(state: np.ndarray, indices: np.ndarray) -> AdArray:
    """Initialize an AdArray for the part of a system not represented by
    :class:`DiagonalAdArray`.

    The returned array has one entry (and one derivative) per entry of ``state``. Only
    the entries at ``indices`` are treated as independent variables (a unit derivative
    with respect to themselves); all other entries get a zero derivative row, since they
    are assumed to be represented elsewhere by a :class:`DiagonalAdArray` (see
    :func:`initialize_diagonal_ad_arrays`).

    Parameters:
        state: The full state vector for this part of the system.
        indices: Indices, into ``state``, of the entries that should be treated as
            independent variables.

    Returns:
        An AdArray with value ``state`` and a diagonal Jacobian that is the identity
        restricted to ``indices``.

    """
    sz = state.size
    derivatives = np.zeros(sz, dtype=float)
    derivatives[indices] = 1.0
    jac = sps.dia_matrix((derivatives, 0), shape=(sz, sz)).tocsr()
    return AdArray(state, jac)


def initAdArrays(variables: list[np.ndarray]) -> list[AdArray]:
    """Initialize a set of AdArrays.

    The variables' gradients will be taken with respect to all variables jointly.

    Parameters:
        variables: A list of numpy arrays, each of which will be represented by an
            AdArray.

    Returns:
        A list of AdArrays, each of which represents one of the variables in the
        ``variables`` list.

    """

    num_values_per_variable = [v.size for v in variables]
    ad_arrays: list[AdArray] = []

    for i, val in enumerate(variables):
        # initiate zero jacobian
        n = num_values_per_variable[i]
        jac = [sps.csc_matrix((n, m)) for m in num_values_per_variable]
        # Set jacobian of variable i to I
        jac[i] = sps.diags(np.ones(num_values_per_variable[i])).tocsr()
        # initiate AdArray
        jac = sps.bmat([jac])
        ad_arrays.append(AdArray(val, jac))

    return ad_arrays


class AdArrayBase(abc.ABC):
    """Common interface of the arrays used for forward mode automatic differentiation.

    An Ad array holds a value, :attr:`val`, together with the derivatives of that value
    with respect to the degrees of freedom of the full system. The concrete classes
    differ in how the derivatives are stored:

      * :class:`AdArray` stores the full Jacobian matrix, as a sparse matrix.
      * :class:`DiagonalAdArray` stores only the diagonal of the Jacobian, which
        suffices for quantities that depend only on themselves, and is cheaper to
        operate on.

    The storage of the derivatives is deliberately not part of this interface: Each
    concrete class has its own attribute ``jac``, and its type and meaning differ
    between the representations. Code that needs the Jacobian should either use
    :attr:`full_jac`, which is available for all Ad arrays, or narrow the type with
    ``isinstance`` before accessing ``jac``.

    Ad arrays implement arithmetic operations with floats, numpy arrays, scipy sparse
    matrices, and other Ad arrays. For these operations, the following general rules
    apply:
      * Scalars can be used for any arithmetic operation except matrix multiplication (
        the @ operator). As a convenience measure to limit the number of cases that must
        be handled and maintained, the scalar must be a float.
      * Numpy arrays are assumed to be 1d and have the same size as the Ad array.
        Numpy arrays can be used for any operation except matrix multiplication.
        The operand order is irrelevant: both ``AdArray + numpy.array`` and
        ``numpy.array + AdArray`` give an Ad array.
      * Scipy matrices can only be used for matrix-vector products (the @ operator), and
        then only for left multiplication. While right multiplication could technically
        work, depending on the size of the matrix, this is not the way the Ad framework
        is intended to be used, and so this operation is not supported.
      * Other Ad arrays, in either representation, can be used with all arithmetic
        operations except the @ operator.

    A violation of these rules will result in a ``ValueError``.

    Operations between Ad arrays in different representations return an
    :class:`AdArray`.

    Attributes:
        val: The value of the Ad array, stored as a 1d numpy array.

    """

    # Turn off numpy's ufuncs for this class, to avoid unexpected behavior when
    # combining with numpy arrays. See
    # https://numpy.org/neps/nep-0013-ufunc-overrides.html#turning-ufuncs-off for
    # technical information and GH issue #819 for a description of the problem that can
    # arise if this is not done.
    __array_ufunc__ = None

    def __init__(self, val: np.ndarray) -> None:
        _check_1d(val)
        # Enforce float format of all data to limit the number of cases we need to
        # handle and test.
        self.val: np.ndarray = _as_float(val)
        """The value of the Ad array, stored as a 1d numpy array."""

    @abc.abstractmethod
    def to_full(self) -> AdArray:
        """Return this Ad array in the full representation.

        Returns:
            An :class:`AdArray` with the same value and Jacobian as this array.

        """

    @property
    def full_jac(self) -> sps.spmatrix | sps.sparray:
        """The Jacobian, as a sparse matrix.

        Available for all representations; arrays that are not stored in the full
        representation are converted on access.

        """
        return self.to_full().jac

    @abc.abstractmethod
    def copy(self, val: Optional[np.ndarray] = None) -> AdArrayBase:
        """Return a copy of this Ad array, optionally with a new value.

        The copy is in the same representation as this array. Its value and derivatives
        are either given, or copied from this array, so that the copy can be modified
        without affecting this array. Structural data that describe the layout of the
        derivatives, if any, are shared with this array; use :meth:`deepcopy` to copy
        these as well.

        Parameters:
            val: Value of the copy. Used as is, without copying. If not given, the value
                of this array is copied.

        Returns:
            A copy of this Ad array.

        """

    @abc.abstractmethod
    def deepcopy(self) -> AdArrayBase:
        """Return a copy of this Ad array, including any structural data.

        Returns:
            A copy of this Ad array which shares no data with it.

        """

    @abc.abstractmethod
    def chain_rule(self, val: np.ndarray, derivative: np.ndarray) -> AdArrayBase:
        """Apply the chain rule for a function evaluated entry by entry on this array.

        For a function ``f`` applied to each entry of this array, ``x``, return
        ``f(x)`` as an Ad array, with the Jacobian ``diag(f'(x)) @ J``, where ``J`` is
        the Jacobian of ``x``.

        Example:
            The exponential function, whose derivative equals its value::

                val = np.exp(x.val)
                y = x.chain_rule(val, val)

        Parameters:
            val: The values ``f(x)``, one per entry in this array.
            derivative: The derivatives ``f'(x)``, one per entry in this array.

        Returns:
            ``f(x)``, in the same representation as this array.

        """

    @abc.abstractmethod
    def __getitem__(
        self, key: slice | np.ndarray[Any, np.dtype[np.int_]]
    ) -> AdArrayBase:
        """Slice the Ad array row-wise."""

    @abc.abstractmethod
    def __setitem__(
        self,
        key: slice | np.ndarray[Any, np.dtype[np.int_]],
        new_value: pp.number | np.ndarray | AdArrayBase,
    ) -> None:
        """Insert new values row-wise."""

    @abc.abstractmethod
    def __add__(self, other: AdType) -> AdArrayBase:
        """Add another object to this Ad array."""

    @abc.abstractmethod
    def __mul__(self, other: AdType) -> AdArrayBase:
        """Elementwise product between this Ad array and another object."""

    @abc.abstractmethod
    def __pow__(self, other: AdType) -> AdArrayBase:
        """Raise this Ad array to the power of another object, elementwise."""

    @abc.abstractmethod
    def __rpow__(self, other: AdType) -> AdArrayBase:
        """Raise another object to the power of this Ad array, elementwise."""

    @abc.abstractmethod
    def __truediv__(self, other: AdType) -> AdArrayBase:
        """Divide this Ad array by another object, elementwise."""

    @abc.abstractmethod
    def __rtruediv__(self, other: AdType) -> AdArrayBase:
        """Divide another object by this Ad array, elementwise."""

    @abc.abstractmethod
    def __rmatmul__(self, other: AdType) -> AdArrayBase:
        """Left-multiply this Ad array by a sparse matrix."""

    def _check_1d_operand(self, other: np.ndarray, op_name: str) -> None:
        """Raise a ValueError unless ``other`` is a one dimensional numpy array.

        Parameters:
            other: The numpy array to check.
            op_name: Name of the operation, used in the error message.

        """
        if other.ndim != 1:
            raise ValueError(f"Only 1d numpy arrays can be used for AdArray {op_name}.")

    def __radd__(self, other: AdType) -> AdArrayBase:
        """Add the AdArray to another object.

        Parameters:
            other: An object to be added to this object. See class documentation for
                restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An AdArray which combines ``self`` and ``other``.

        """
        return self.__add__(other)

    def __sub__(self, other: AdType) -> AdArrayBase:
        """Subtract right hand operand (this AdArray) from left hand operand (other).

        Parameters:
            other: An object to be subtracted from this object. See class
            documentation for restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An AdArray which combines ``self`` and ``other``.

        """
        return self.__add__(-other)

    def __rsub__(self, other: AdType) -> AdArrayBase:
        """Subtract right hand operand (other) from left hand operand (this AdArray).

        Parameters:
            other: An object to be subtracted from this object. See class
            documentation for restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An AdArray which subtracts ``self`` from ``other``.

        """
        # Calculate self - other and negative the answer (note the minus sign in front).
        return -self.__sub__(other)

    def __rmul__(self, other: AdType) -> AdArrayBase:
        """Elementwise product (Hadamard or Schur product) between two objects.

        Parameters:
            other: An object to be multiplied with this object. See class documentation
                for restrictions on admissible types for this function.

        Returns:
            An AdArray which multiplies ``self`` and ``other`` elementwise.

        Raises:
            ValueError: If this represents an impermissible operation.

        """

        if isinstance(other, (float, int, np.ndarray, *_SPARSE_TYPES)):
            # In these cases, there is no difference between left and right
            # multiplication, so we simply invoke the standard __mul__ function.
            return self.__mul__(other)

        elif isinstance(other, AdArrayBase):
            # The only way we can end up here is if other.__mul__(self) returns
            # NotImplemented, which makes no sense. Raise an error; if we ever end
            # up here, something is really wrong.
            raise RuntimeError(
                "Something went wrong when multiplying two AdArrays elementwise."
            )
        else:
            raise ValueError(
                f"Unknown type {type(other)} for AdArray elementwise multiplication."
            )

    def __matmul__(self, other: AdType) -> AdArrayBase:
        """The operation `AdArray @ Anything` is disallowed.

        Parameters:
            other: An object which should be right multiplied with this AdArray. See
                class documentation for restrictions on admissible types for this
                function.

        Returns:
            An AdArray which represents ``self`` @ ``other`` elementwise.

        """

        if isinstance(other, (int, float, np.ndarray, AdArrayBase)):
            raise ValueError(
                """Cannot perform matrix multiplication between an AdArray and a"""
                f""" {type(other)}."""
            )

        elif isinstance(other, _SPARSE_TYPES):
            # This goes against the way equations should be formulated in the AD
            # framework, variables should not be right-multiplied by anything. Raise a
            # value error to make sure this is not done.
            raise ValueError(
                """AdArrays should only be left-multiplied by sparse matrices."""
            )

        else:
            raise ValueError(f"Unknown type {type(other)} for AdArray multiplication.")

    def __neg__(self) -> AdArrayBase:
        return self * -1.0

    def __lt__(self, other: AdType) -> bool | np.ndarray:
        """Overload of operation ``self < other``.

        The Ad-array delegates the logical operation solely to the values :attr:`val`,
        leaving the actual implementation to numpy.
        I.e., any binary, logical operation is equivalent to what numpy does with the
        values.

        Parameters:
            other: Right-hand side operand. If it is an Ad-array, its :attr:`val` is
                used to invoke the overload of numpy.

        Returns:
            A boolean (array) as the result of the lesser-operation.

        """
        if isinstance(other, AdArrayBase):
            return self.val < other.val
        else:
            return self.val < other

    def __le__(self, other: AdType) -> bool | np.ndarray:
        """Overload for ``self <= other``. See :meth:`__lt__` for more information."""
        if isinstance(other, AdArrayBase):
            return self.val <= other.val
        else:
            return self.val <= other

    def __gt__(self, other: AdType) -> bool | np.ndarray:
        """Overload for ``self > other``. See :meth:`__lt__` for more information."""
        if isinstance(other, AdArrayBase):
            return self.val > other.val
        else:
            return self.val > other

    def __ge__(self, other: AdType) -> bool | np.ndarray:
        """Overload for ``self >= other``. See :meth:`__lt__` for more information."""
        if isinstance(other, AdArrayBase):
            return self.val >= other.val
        else:
            return self.val >= other

    def __eq__(self, other: AdType) -> bool | np.ndarray:  # type:ignore[override]
        """Overload for ``self == other``. See :meth:`__lt__` for more information."""
        # mypy complaints that parent class object returns only bool here.
        # But we leave the equal operation to the numpy values.
        if isinstance(other, AdArrayBase):
            return self.val == other.val
        else:
            return self.val == other

    def __ne__(self, other: AdType) -> bool | np.ndarray:  # type:ignore[override]
        """Overload for ``self != other``. See :meth:`__lt__` for more information."""
        # NOTE without the override of __ne__, Python uses __eq__ and returns its
        # negation. In the scalar case (val.shape = (1,)) this can return a boolean,
        # not a boolean array with shape (1,)
        if isinstance(other, AdArrayBase):
            return self.val != other.val
        else:
            return self.val != other


class AdArray(AdArrayBase):
    """An Ad array with the Jacobian stored as a sparse matrix.

    This is the general representation, which can hold any Jacobian. See
    :class:`AdArrayBase` for the rules for arithmetic operations, and
    :class:`DiagonalAdArray` for a compact representation of quantities that depend only
    on themselves.

    Parameters:
        val: The value of the Ad array, as a 1d numpy array.
        jac: The Jacobian matrix, as a sparse matrix with one row per entry in ``val``.

    Raises:
        TypeError: If ``jac`` is not a sparse matrix.
        ValueError: If ``val`` is not 1d, or if the number of rows in ``jac`` does not
            match the size of ``val``.

    Attributes:
        val: The value of the AdArray, stored as a 1d numpy array.
        jac: The Jacobian matrix of the AdArray, stored as a sparse matrix.

    """

    def __init__(self, val: np.ndarray, jac: sps.spmatrix | sps.sparray) -> None:
        # Consistency checks, to limit the possibilities for errors when combining this
        # array with other objects.
        super().__init__(val)
        if not isinstance(jac, _SPARSE_TYPES):
            # A dense Jacobian is most likely the diagonal representation used by
            # DiagonalAdArray, which this class cannot interpret.
            raise TypeError(
                f"The Jacobian of an AdArray must be a sparse matrix, not {type(jac)}"
            )
        if jac.shape[0] != self.val.size:
            raise ValueError(
                "The Jacobian matrix should have one row per array degree of freedom"
            )
        self.jac: sps.spmatrix | sps.sparray = _as_float(jac)
        """The Jacobian matrix of the AdArray, stored as a sparse matrix."""

    def to_full(self) -> AdArray:
        """Return this AdArray, which is already in the full representation.

        Returns:
            This AdArray.

        """
        return self

    def copy(
        self,
        val: Optional[np.ndarray] = None,
        jac: Optional[sps.spmatrix | sps.sparray] = None,
    ) -> AdArray:
        """Return a copy of this AdArray, optionally with a new value and Jacobian.

        Parameters:
            val: Value of the copy. Used as is, without copying. If not given, the value
                of this array is copied.
            jac: Jacobian of the copy, as a sparse matrix. Used as is, without copying.
                If not given, the Jacobian of this array is copied.

        Returns:
            A copy of this AdArray.

        """
        return AdArray(
            self.val.copy() if val is None else val,
            self.jac.copy() if jac is None else jac,
        )

    def deepcopy(self) -> AdArray:
        """Return a copy of this AdArray.

        An AdArray has no structural data beyond its value and Jacobian, hence this is
        the same as :meth:`copy` without arguments.

        Returns:
            A copy of this AdArray which shares no data with it.

        """
        return self.copy()

    def chain_rule(self, val: np.ndarray, derivative: np.ndarray) -> AdArray:
        """Apply the chain rule for a function evaluated entry by entry on this array.

        For a function ``f`` applied to each entry of this array, ``x``, return
        ``f(x)`` as an Ad array, with the Jacobian ``diag(f'(x)) @ J``, where ``J`` is
        the Jacobian of ``x``.

        Example:
            The exponential function, whose derivative equals its value::

                val = np.exp(x.val)
                y = x.chain_rule(val, val)

        Parameters:
            val: The values ``f(x)``, one per entry in this array.
            derivative: The derivatives ``f'(x)``, one per entry in this array.

        Returns:
            ``f(x)``, in the same representation as this array.

        """
        return AdArray(val, self.diagvec_mul_jac(derivative))

    def diagvec_mul_jac(self, a: np.ndarray) -> sps.spmatrix:
        """Left-multiply the Jacobian by a diagonal matrix represented as a vector.

        Parameters:
            a: The diagonal entries of the (implicit) diagonal matrix to
                left-multiply the Jacobian with.

        Returns:
            The product, as a sparse matrix.

        """
        return sps.diags(a) * self.jac

    if TYPE_CHECKING:
        # The implementations of these operations are shared with DiagonalAdArray, see
        # AdArrayBase. They delegate to the operations implemented below, and so return
        # an AdArray when called on one. Declare this for the type checker.
        def __radd__(self, other: AdType) -> AdArray: ...

        def __sub__(self, other: AdType) -> AdArray: ...

        def __rsub__(self, other: AdType) -> AdArray: ...

        def __rmul__(self, other: AdType) -> AdArray: ...

        def __neg__(self) -> AdArray: ...

    def __str__(self) -> str:
        s = f"Ad array of size {self.val.size}\n"
        s += f"Jacobian is of size {self.jac.shape} and has {self.jac.data.size}"
        s += " elements."
        return s

    def __repr__(self) -> str:
        s = f"Ad array of size {self.val.size}\n"
        s += f"Value: {self.val}\n"
        s += f"Jacobian: {self.jac}"
        return s

    def __getitem__(self, key: slice | np.ndarray[Any, np.dtype[np.int_]]) -> AdArray:
        """Slice the Ad Array row-wise (value and Jacobian).

        Parameters:
            key: A row-index (integer) or slice object to be applied to :attr:`val` and
                :attr:`jac`

        Returns:
            A new Ad array with values and Jacobian sliced row-wise.

        """
        # NOTE mypy complains even though numpy arrays can handle slices [x:y:z]
        # Probably a missing type annotation on numpy's side
        val = self.val[key]  # type:ignore[index]
        # in case of single index, broadcast to 1D array
        if val.ndim == 0:
            val = np.array([val])
        return AdArray(val, self.jac[key])

    def __setitem__(
        self,
        key: slice | np.ndarray[Any, np.dtype[np.int_]],
        new_value: pp.number | np.ndarray | AdArrayBase,
    ) -> None:
        """Insert new values in :attr:`val` and :attr:`jac` row-wise.

        Note:
            Broadcasting is outsourced to numpy and scipy. If ``new_value`` is not
                compatible in terms of size and ``key``, respective errors are raised.

        Parameters:
            key: A row-index (integer) or slice object to set the rows in value and
                Jacobian
            new_value: New values for :attr:`val` and rows of :attr:`jac`.
                If ``new_value`` is an Ad array, its Jacobian, in the full
                representation, is inserted into the defined rows.

        Raises:
            NotImplementedError: If ``new_value`` is not a number, numpy array or
                Ad array.

        """
        if isinstance(new_value, np.ndarray | pp.number):
            self.val[key] = new_value
        elif isinstance(new_value, AdArrayBase):
            new_value = new_value.to_full()
            self.val[key] = new_value.val
            self.jac[key] = new_value.jac
        else:
            raise NotImplementedError("Setting")

    def _prepare_other_ad(self, other: AdArrayBase, op_name: str) -> AdArray:
        """Convert ``other`` to the full representation and validate that its size and
        Jacobian shape are compatible with this array, ahead of a binary operation
        between two Ad arrays.

        Parameters:
            other: The other AdArray in the operation.
            op_name: Name of the operation, used in the error message if the sizes are
                incompatible.

        Raises:
            ValueError: If the sizes of the two arrays are incompatible.

        Returns:
            ``other``, in the full representation.

        """
        other = other.to_full()
        if self.val.size != other.val.size or self.jac.shape != other.jac.shape:
            raise ValueError(f"Incompatible sizes for AdArray {op_name}.")
        return other

    def __add__(self, other: AdType) -> AdArray:
        """Add the AdArray to another object.

        Parameters:
            other: An object to be added to this object. See class documentation for
                restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An AdArray which combines ``self`` and ``other``.

        """
        if isinstance(other, (int, float)):
            # Strictly speaking, we require scalars to be floats, but add casting of
            # ints to floats for convenience.
            return AdArray(self.val + float(other), self.jac)

        elif isinstance(other, np.ndarray):
            self._check_1d_operand(other, "addition")
            return AdArray(self.val + other, self.jac)

        elif isinstance(other, _SPARSE_TYPES):
            raise ValueError("Sparse matrices cannot be added to AdArrays")

        elif isinstance(other, AdArrayBase):
            other = self._prepare_other_ad(other, "addition")
            return AdArray(self.val + other.val, self.jac + other.jac)
        else:
            raise ValueError(f"Unknown type {type(other)} for AdArray addition")

    def __mul__(self, other: AdType) -> AdArray:
        """Elementwise product (Hadamard or Schur product) between two objects.

        Parameters:
            other: An object to be multiplied with this object. See class documentation
                for restrictions on admissible types for this function.

        Raises:
            ValueError: If this represents an impermissible operation.

        Returns:
            An AdArray which multiplies ``self`` and ``other`` elementwise.

        """
        # Use if-else with isinstance to identify the other operator.
        if isinstance(other, (int, float)):
            # Strictly speaking, we require scalars to be floats, but add casting of
            # ints to floats for convenience.
            return AdArray(self.val * other, self.jac * other)

        elif isinstance(other, np.ndarray):
            self._check_1d_operand(other, "elementwise multiplication")
            # The below line will invoke numpy's __mul__ method on the values.
            new_val = self.val * other
            # The Jacobian will have its columns scaled with the values in other.
            # Achieve this by left-multiplying with other, represented as a diagonal
            # matrix.
            new_jac = self.diagvec_mul_jac(other)
            return AdArray(new_val, new_jac)

        elif isinstance(other, _SPARSE_TYPES):
            raise ValueError(
                """Sparse matrices cannot be multiplied with  AdArrays elementwise.
                Did you mean to use the @ operator?
                """
            )

        elif isinstance(other, AdArrayBase):
            other = self._prepare_other_ad(other, "elementwise multiplication")

            # For the values, use elementwise multiplication, as implemented by
            # numpy's __mul__ method
            new_val = self.val * other.val
            # Compute the derivative of the product using the product rule. Since
            # the gradients in jac is stored row-wise, the columns in self.jac
            # should be scaled with the values of other and vice versa.
            new_jac = self.diagvec_mul_jac(other.val) + other.diagvec_mul_jac(self.val)
            return AdArray(new_val, new_jac)

        elif isinstance(other, pp.matrix_operations.ArraySlicer):
            return other.__rmul__(self)

        else:
            raise ValueError(
                f"Unknown type {type(other)} for AdArray elementwise multiplication."
            )

    def __pow__(self, other: AdType) -> AdArray:
        """Raise this AdArray to the power of another object.

        Parameters:
            other: An object with exponent to which this AdArray is raised. The power
                is implemented elementwise. See class documentation for restrictions on
                admissible types for this function.

        Returns:
            An AdArray which represents ``other`` ** ``self`` elementwise.

        """

        if isinstance(other, (int, float)):
            # This is a polynomial, use standard rules for differentiation.
            new_val = self.val**other
            # Left-multiply jac with a diagonal-matrix version of the differentiated
            # polynomial, this will give the desired column-wise scaling of the
            # gradients.
            new_jac = self.diagvec_mul_jac(float(other) * self.val ** float(other - 1))
            return AdArray(new_val, new_jac)

        elif isinstance(other, np.ndarray):
            self._check_1d_operand(other, "power")
            # This is a polynomial, but with different coefficients for each element
            # in self.val. Numpy can be picky on raising arrays to negative powers,
            # without EK ever understanding why, so we convert to a float
            # beforehand, just to be sure.
            new_val = self.val ** other.astype(float)
            # The Jacobian will have its columns scaled with the values in other,
            # again in array-form. Achieve this by left-multiplying with other,
            # represented as a diagonal matrix.
            new_jac = self.diagvec_mul_jac(other * (self.val ** (other - 1)))
            return AdArray(new_val, new_jac)

        elif isinstance(other, _SPARSE_TYPES):
            raise ValueError("Cannot raise AdArrays to power of sparse matrices.")

        elif isinstance(other, pp.matrix_operations.ArraySlicer):
            return other.__rpow__(self)

        elif isinstance(other, AdArrayBase):
            other = self._prepare_other_ad(other, "power")

            # This is an expression of the type f = x^y, with derivative
            #
            #   df = (y * x ** (y-1)) * dx + x^y * log(x) * dy
            #
            # Compute the value using numpy's power method. Convert to float to
            # avoid spurious behavior form numpy, just to be sure.
            new_val = self.val ** other.val.astype(float)
            # The derivative, computed by the chain rule.
            new_jac = self.diagvec_mul_jac(
                other.val * self.val ** (other.val.astype(float) - 1.0)
            ) + other.diagvec_mul_jac(
                self.val ** other.val.astype(float) * np.log(self.val)
            )

            return AdArray(new_val, new_jac)

        else:
            raise ValueError(f"Unknown type {type(other)} for AdArray power.")

    def __rpow__(self, other: AdType) -> AdArray:
        """Raise another object to the power of this AdArray.

        Parameters:
            other: An object which should be raised to the power of this AdArray.
                The power is implemented elementwise. See class documentation for
                restrictions on admissible types for this function.

        Returns:
            An AdArray which represents ``other`` ** ``self`` elementwise.

        """
        if isinstance(other, (int, float)):
            # This is an exponent of type number ** x
            new_val = float(other) ** self.val
            # Left-multiply jac with a diagonal-matrix version of the differentiated
            # polynomial, this will give the desired column-wise scaling of the
            # gradients.
            new_jac = self.diagvec_mul_jac(
                (float(other) ** self.val) * np.log(float(other))
            )
            return AdArray(new_val, new_jac)

        elif isinstance(other, np.ndarray):
            self._check_1d_operand(other, "power")
            # This is an exponent with different coefficients for each element in
            # self.val. Numpy appears to be using dtype instead of values to determine
            # the output type. Consequently, the multiplicative inverse / negative
            # integer powers of a numpy's integer array lead to a float array raising a
            # value error. As an example compare 1/np.array([1,2,3]) and
            # np.array([1,2,3])**-1. As a workaround, we convert it to a float.
            new_val = other.astype(float) ** self.val
            # The Jacobian will have its columns scaled with the values in other, again
            # in array-form. Achieve this by left-multiplying with other, represented as
            # a diagonal matrix.
            new_jac = self.diagvec_mul_jac((other**self.val) * np.log(other))
            return AdArray(new_val, new_jac)

        elif isinstance(other, _SPARSE_TYPES):
            raise ValueError("Cannot raise sparse matrices to the power of Ad arrays.")

        elif isinstance(other, AdArrayBase):
            other = self._prepare_other_ad(other, "power")
            return other.__pow__(self)

        else:
            raise ValueError(f"Unknown type {type(other)} for AdArray power.")

    def __truediv__(self, other: AdType) -> AdArray:
        """Divide this AdArray by another object.

        Parameters:
            other: An object which should divide this AdArray. The division is
                implemented elementwise. See class documentation for restrictions on
                admissible types for this function.

        Returns:
            An AdArray which represents ``self`` / ``other`` elementwise.

        """

        if isinstance(other, (int, float)):
            # Division by float, or int cast to float is straightforward, elementwise.
            new_val = self.val / float(other)
            new_jac = self.jac / float(other)
            return AdArray(new_val, new_jac)

        elif isinstance(other, np.ndarray):
            self._check_1d_operand(other, "division")

            new_val = self.val * other.astype(float) ** (-1.0)
            # The Jacobian will have its columns scaled with the values in other, again
            # in array-form. Achieve this by left-multiplying with other, represented as
            # a diagonal matrix.
            new_jac = self.diagvec_mul_jac(other.astype(float) ** (-1.0))
            return AdArray(new_val, new_jac)

        elif isinstance(other, _SPARSE_TYPES):
            raise ValueError("AdArrays cannot be divided by sparse matrices.")

        elif isinstance(other, pp.matrix_operations.ArraySlicer):
            return other.__rtruediv__(self)

        elif isinstance(other, AdArrayBase):
            other = self._prepare_other_ad(other, "division")
            return self.__mul__(other.__pow__(-1.0))

        else:
            raise ValueError(f"Unknown type {type(other)} for AdArray division.")

    def __rtruediv__(self, other: AdType) -> AdArray:
        """Divide another object by this AdArray.

        Parameters:
            other: An object which should be divided by this AdArray. The division is
                implemented elementwise. See class documentation for restrictions on
                admissible types for this function.

        Returns:
            An AdArray which represents ``other`` / ``self`` elementwise.

        """

        if isinstance(other, (float, int, np.ndarray, *_SPARSE_TYPES)):
            # Divide a float or a numpy array by self is the same as raising self to the
            # power of -1 and multiplying by the float. The multiplication will end up
            # calling self.__mul__, which will do the right checks for numpy arrays and
            # sparse matrices.
            return self.__pow__(-1.0) * other

        elif isinstance(other, AdArrayBase):
            other = self._prepare_other_ad(other, "division")
            return other.__mul__(self.__pow__(-1.0))

        else:
            raise ValueError(f"Unknown type {type(other)} for AdArray division.")

    def __rmatmul__(self, other: AdType) -> AdArray:
        """Do a matrix multiplication between another object and this AdArray.

        Parameters:
            other: An object which should be left multiplied with this AdArray. See
                class documentation for restrictions on admissible types for this
                function.

        Returns:
            An AdArray which represents ``other`` @ ``self``.

        """
        if isinstance(other, (int, float, np.ndarray, AdArrayBase)):
            raise ValueError(
                """Cannot perform matrix multiplication between an AdArray and a"""
                f""" {type(other)}."""
            )

        elif isinstance(other, _SPARSE_TYPES):
            # This is the standard matrix-vector multiplication.
            if self.jac.shape[0] != other.shape[1]:
                raise ValueError(
                    """Dimension mismatch between sparse matrix and AdArray during
                    matrix multiplication."""
                )
            new_val = other @ self.val
            new_jac = other @ self.jac
            return AdArray(new_val, new_jac)

        else:
            raise ValueError(f"Unknown type {type(other)} for AdArray multiplication.")


def initialize_diagonal_ad_arrays(
    variables: list[np.ndarray],
    indices: list[np.ndarray],
    num_derivatives: int,
    derivatives: list[np.ndarray] | None = None,
) -> list[DiagonalAdArray]:
    """Initialize a set of DiagonalAdArrays, each depending only on itself.

    Each returned array has, by default, a unit derivative with respect to its own
    entries and no dependence on the other variables in ``variables``.

    Parameters:
        variables: A list of numpy arrays, each of which will be represented by a
            DiagonalAdArray.
        indices: For each array in ``variables``, the indices, into the full system
            of ``num_derivatives`` degrees of freedom, of that array's entries.
        num_derivatives: Total number of derivatives (degrees of freedom) in the
            full system.
        derivatives: If provided, the diagonal Jacobian entries to use for each
            variable, instead of the default of 1.0 (a unit derivative). Used e.g. to
            represent the derivative of a SurrogateOperator with respect to its primary
            variable.

    Returns:
        A list of DiagonalAdArrays, each of which represents one of the variables in the
        ``variables`` list.

    Raises:
        ValueError: If the number of ``variables`` and ``indices`` do not match, or if
            the size of an array in ``variables`` does not match the size of the
            corresponding array in ``indices``.

    """
    if len(variables) != len(indices):
        raise ValueError("Number of variables should match number of offsets.")
    for var, ind in zip(variables, indices):
        if var.size != ind.size:
            raise ValueError("Number of variables should match number of indices.")

    num_vars = len(variables)
    sz_vars = variables[0].size if len(variables) > 0 else 0

    diagonal_variables = []

    for variable_index in range(num_vars):
        val = variables[variable_index]
        jac = np.zeros((num_vars, sz_vars))
        if derivatives is not None:
            jac[variable_index] = derivatives[variable_index]
        else:
            jac[variable_index] = 1.0
        diagonal_variables.append(
            DiagonalAdArray(val, jac, indices[variable_index], indices, num_derivatives)
        )
    return diagonal_variables


class DiagonalAdArray(AdArrayBase):
    """An Ad array where only the diagonal of the Jacobian is stored.

    This representation can be used for quantities which only depend on themselves, and
    not on any other variables. The operations are implemented in a way that they take
    advantage of this structure to speed up calculations. Operations whose result cannot
    be represented this way, such as left-multiplication with a sparse matrix, return an
    :class:`AdArray`.

    The derivatives are stored as a 2d numpy array, :attr:`jac`, with one row per block
    of the Jacobian and one column per entry in :attr:`val`. Together with the
    structural information in :attr:`row_indices`, :attr:`col_indices` and
    :attr:`num_derivatives`, this suffices to construct the full Jacobian, see
    :meth:`to_full`.

    Parameters:
        val: The value of the Ad array, as a 1d numpy array.
        jac: The diagonals of the Jacobian, as a 2d numpy array with one row per block
            of the Jacobian and one column per entry in ``val``. A 1d array is
            interpreted as a single block.
        row_indices: Indices, into the full system of ``num_derivatives`` degrees of
            freedom, of the entries in ``val``.
        col_indices: For each block of the Jacobian, the indices, into the full system,
            of the columns of the full Jacobian that the entries in the block belong to.
        num_derivatives: Total number of derivatives (degrees of freedom) in the full
            system, that is, the number of columns of the full Jacobian.

    Raises:
        ValueError: If ``val`` is not 1d, or if the number of columns in ``jac`` does
            not match the size of ``val``.

    """

    def __init__(
        self,
        val: np.ndarray,
        jac: np.ndarray,
        row_indices: np.ndarray,
        col_indices: list[np.ndarray],
        num_derivatives: int,
    ) -> None:
        super().__init__(val)
        if jac.ndim == 1:
            jac = jac[np.newaxis, :]
        if jac.shape[1] != self.val.size:
            raise ValueError(
                "The diagonal Jacobian should have one column per array degree of "
                "freedom"
            )
        self.jac: np.ndarray = _as_float(jac)
        """The derivatives, stored as one row per block of the Jacobian."""

        self._num_derivatives = num_derivatives
        """Total number of derivatives in the system."""
        self._row_indices = row_indices
        self._col_indices = col_indices

    @property
    def row_indices(self) -> np.ndarray:
        """The row indices of this array's entries in the full, non-diagonal
        representation of the Jacobian."""
        return self._row_indices

    @property
    def col_indices(self) -> list[np.ndarray]:
        """The column indices of this array's entries in the full, non-diagonal
        representation of the Jacobian."""
        return self._col_indices

    @property
    def num_derivatives(self) -> int:
        """Total number of derivatives in the system."""
        return self._num_derivatives

    def to_full(self) -> AdArray:
        """Convert this DiagonalAdArray to a full AdArray, where the Jacobian is stored
        as a sparse matrix.

        Returns:
            An AdArray with the same value and Jacobian as this DiagonalAdArray, but
            with the Jacobian stored as a sparse matrix.

        """
        num_vars = self.jac.shape[0]

        num_indices = self._row_indices.size

        indptr = np.arange(0, num_indices * num_vars + 1, num_vars)
        indices = np.vstack([col for col in self._col_indices]).ravel("F")
        jac = sps.csr_matrix(
            (self.jac.ravel("F"), indices, indptr),
            shape=(num_indices, self._num_derivatives),
        )
        return AdArray(self.val, jac)

    def copy(
        self, val: Optional[np.ndarray] = None, jac: Optional[np.ndarray] = None
    ) -> DiagonalAdArray:
        """Return a copy of this DiagonalAdArray, optionally with a new value and
        Jacobian.

        The copy shares the structural data, :attr:`row_indices`, :attr:`col_indices`
        and :attr:`num_derivatives`, with this array; use :meth:`deepcopy` to copy these
        as well.

        Parameters:
            val: Value of the copy. Used as is, without copying. If not given, the value
                of this array is copied.
            jac: Jacobian of the copy, in the diagonal representation. Used as is,
                without copying. If not given, the Jacobian of this array is copied.

        Returns:
            A copy of this DiagonalAdArray.

        """
        return DiagonalAdArray(
            self.val.copy() if val is None else val,
            self.jac.copy() if jac is None else jac,
            self._row_indices,
            self._col_indices,
            self._num_derivatives,
        )

    def deepcopy(self) -> DiagonalAdArray:
        """Return a copy of this DiagonalAdArray, including its structural data.

        Returns:
            A copy of this DiagonalAdArray which shares no data with it.

        """
        return DiagonalAdArray(
            self.val.copy(),
            self.jac.copy(),
            self._row_indices.copy(),
            [col_ind.copy() for col_ind in self._col_indices],
            self._num_derivatives,
        )

    def chain_rule(self, val: np.ndarray, derivative: np.ndarray) -> DiagonalAdArray:
        """Apply the chain rule for a function evaluated entry by entry on this array.

        For a function ``f`` applied to each entry of this array, ``x``, return
        ``f(x)`` as an Ad array, with the Jacobian ``diag(f'(x)) @ J``, where ``J`` is
        the Jacobian of ``x``.

        Example:
            The exponential function, whose derivative equals its value::

                val = np.exp(x.val)
                y = x.chain_rule(val, val)

        Parameters:
            val: The values ``f(x)``, one per entry in this array.
            derivative: The derivatives ``f'(x)``, one per entry in this array.

        Returns:
            ``f(x)``, in the same representation as this array.

        """
        # Each row of jac holds one block of derivatives, one column per entry, so the
        # scaling by derivative broadcasts over the rows.
        return self.copy(val, derivative * self.jac)

    def __str__(self) -> str:
        s = f"Diagonal Ad array of size {self.val.size}\n"
        s += f"Jacobian of size {(self.val.size, self._num_derivatives)} is stored as "
        s += f"{self.jac.shape[0]} diagonal block(s)."
        return s

    def __repr__(self) -> str:
        s = f"Diagonal Ad array of size {self.val.size}\n"
        s += f"Value: {self.val}\n"
        s += f"Jacobian, diagonal blocks: {self.jac}"
        return s

    def __getitem__(
        self, key: slice | np.ndarray[Any, np.dtype[np.int_]]
    ) -> DiagonalAdArray:
        """Return a new array with the specified entries.

        Parameters:
            key: Slice or index array selecting the entries to set.

        Returns:
            A new DiagonalAdArray with the specified entries of :attr:`val` and
            :attr:`jac`, and the corresponding structural indices.

        """
        vals = self.val[key]
        jac = self.jac[:, key]

        row_indices = self._row_indices[key]
        col_indices = [col_ind[key] for col_ind in self._col_indices]
        return DiagonalAdArray(
            vals, jac, row_indices, col_indices, self._num_derivatives
        )

    def __setitem__(
        self,
        key: slice | np.ndarray[Any, np.dtype[np.int_]],
        new_value: pp.number | np.ndarray | AdArrayBase,
    ) -> None:
        """Insert new values in :attr:`val` and the columns of :attr:`jac`.

        Parameters:
            key: Slice or index array selecting the entries to set.
            new_value: New values. A number or numpy array only replaces the values; the
                derivatives are left unchanged. A DiagonalAdArray replaces both. It is
                assumed to share the structure of this array, that is, to have the same
                number of Jacobian blocks and matching row and column indices.

        Raises:
            NotImplementedError: If ``new_value`` is an Ad array in the full
                representation, which cannot be inserted into a diagonal one, or of any
                other unsupported type.

        """
        if isinstance(new_value, np.ndarray | pp.number):
            self.val[key] = new_value
        elif isinstance(new_value, DiagonalAdArray):
            # Set the derivatives first: If the shapes are incompatible, this fails
            # before anything has been modified.
            self.jac[:, key] = new_value.jac
            self.val[key] = new_value.val
        elif isinstance(new_value, AdArrayBase):
            raise NotImplementedError(
                "Cannot insert an Ad array in the full representation into a "
                "DiagonalAdArray. Convert the target with to_full() first."
            )
        else:
            raise NotImplementedError("Setting")

    def __add__(self, other: AdType) -> AdArrayBase:
        if isinstance(other, (float, int, np.ndarray)):
            return self.copy(self.val + other, self.jac)
        elif isinstance(other, DiagonalAdArray):
            val = self.val + other.val
            jac = self.jac + other.jac
            return self.copy(val, jac)
        else:
            return self.to_full().__add__(other)

    def __mul__(self, other: AdType) -> AdArrayBase:
        if isinstance(other, (float, int, np.ndarray)):
            return self.copy(self.val * other, self.jac * other)
        elif isinstance(other, DiagonalAdArray):
            val = self.val * other.val
            # Row-broadcast against self.jac/other.jac (shape (num_blocks, n)).
            jac = (
                self.jac * other.val[np.newaxis, :]
                + other.jac * self.val[np.newaxis, :]
            )
            return self.copy(val, jac)

        else:
            return self.to_full().__mul__(other)

    def __pow__(self, other: AdType) -> AdArrayBase:
        if isinstance(other, (float, int, np.ndarray)):
            val = self.val**other
            jac = other * self.val ** (other - 1) * self.jac
            return self.copy(val, jac)
        elif isinstance(other, DiagonalAdArray):
            val = self.val**other.val
            jac = (
                other.val * self.val ** (other.val - 1) * self.jac
                + self.val**other.val * np.log(self.val) * other.jac
            )
            return self.copy(val, jac)

        else:
            return self.to_full().__pow__(other)

    def __rpow__(self, other: AdType) -> AdArrayBase:
        if isinstance(other, (float, int, np.ndarray)):
            val = other**self.val
            jac = (other**self.val) * np.log(other) * self.jac
            return self.copy(val, jac)
        elif isinstance(other, DiagonalAdArray):
            return other.__pow__(self)
        else:
            return self.to_full().__rpow__(other)

    def __truediv__(self, other: AdType) -> AdArrayBase:
        if isinstance(other, (float, int, np.ndarray)):
            val = self.val / other
            jac = self.jac / other
            return self.copy(val, jac)
        elif isinstance(other, DiagonalAdArray):
            val = self.val / other.val
            jac = (self.jac * other.val - self.val * other.jac) / (other.val**2)
            return self.copy(val, jac)
        else:
            return self.to_full().__truediv__(other)

    def __rtruediv__(self, other: AdType) -> AdArrayBase:
        if isinstance(other, (float, int, np.ndarray)):
            val = other / self.val
            jac = -other * self.jac / (self.val**2)
            return self.copy(val, jac)
        elif isinstance(other, DiagonalAdArray):
            return other.__truediv__(self)
        else:
            return self.to_full().__rtruediv__(other)

    def __rmatmul__(self, other: AdType) -> AdArray:
        # When multiplying with a sparse matrix, the expectation is that the result will
        # no longer be suitable for a diagonal representation, so we convert to a full
        # AdArray and perform the multiplication there. The only (reasonably simple)
        # potential simplifications would be if the matrix is either diagonal or a
        # permutation matrix, but we do not expect these cases to be common enough to
        # warrant a special implementation.
        return self.to_full().__rmatmul__(other)
