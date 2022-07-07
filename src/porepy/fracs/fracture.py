"""A module containing the abstract base class for all fractures."""

from __future__ import annotations

import abc
from typing import Generator, List, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike

import porepy as pp
from porepy.utils import setmembership

FractureType = TypeVar("FractureType", bound="Fracture")


class Fracture(abc.ABC):
    """Abstract base class for representing a single fracture.

    This base class provides general functionalities agnostic to the dimension of
    the fracture and the ambient dimension. It contains various utility methods,
    mainly intended for the use together with the FractureNetwork class.

    A fracture is defined by its `npt` vertices, stored in an `nd x npt` numpy-array,
    where `nd` is the assumed ambient dimension. The order/sorting of vertices has to be
    implemented explicitly.

    Dimension-dependent routines are implemented as abstract methods.

    PorePy currently only supports planar and convex fractures fully.
    As a work-around, the fracture can be split into convex parts.

    Attributes:
        points (ArrayLike, nd x npt): Fracture vertices, stored in the implemented order.
        orig_pts (np.ndarray, nd x npt): Original fracture vertices, kept in case the fracture
            geometry is modified.
        center (np.ndarray, nd x 1): Fracture centroid.
        normal (np.ndarray, nd x 1): Normal vector.
        index (int): Index of fracture. Intended use in FractureNetwork. Exact use is not
            clear (several fractures can be given same index), use it with care.

    """

    def __init__(
        self,
        points: ArrayLike,
        index: Optional[int] = None,
        check_convexity: bool = False,
        sort_points: bool = True,
    ):
        """Instatiation of the abstract fracture class.

        The constructor sets the vertices of the fracture and computes the centroid and
        normals. Optionally, checks for planarity and convexity are performed.

        Args:
            points (ArrayLike, nd x npt): Vertices of new fracture. How many are necessary,
                depends on the dimension. Note that `points` is an np.ndarray or any suitable
                object that can be converted into an np.ndarray via np.asarray(`points`). See,
                e.g.: https://numpy.org/devdocs/reference/typing.html#arraylike
            index (int, optional): Index of fracture. Defaults to None.
            check_convexity (bool, optional):  Defaults to False. If True, check if the
                given points are convex. Can be skipped if it is known to be convex to save
                (substantial) time.
            sort_points (bool , optional): Defaults to True. If True, sorts the points
                according to the implemented order. Convexity may play a role, thus if a
                fracture is known to be non-convex, the parameter should be False.

        Raises:
            AssertionError: Asserts the fracture is planar (PorePy currently only fully
                supports planar fractures).
            AssertionError: (Optionally) Asserts the fracture is convex (PorePy currently
                only fully supports convex fractures).

        """
        self.pts = np.asarray(
            points, dtype=np.float64
        )  # note that `points` will mutate
        self._check_pts()

        # Ensure that the points are sorted
        if sort_points:
            self.sort_points()

        self.normal: np.ndarray = self.compute_normal()
        self.center: np.ndarray = self.compute_centroid()
        self.orig_pts: np.ndarray = self.pts.copy()
        self.index = index

        assert self.is_planar(), "Points define non-planar fracture"
        if check_convexity:
            assert self.is_convex(), "Points form non-convex polygon"

    def __repr__(self) -> str:
        """Representation is same as str-representation."""
        return self.__str__()

    def __str__(self) -> str:
        """The str-representation displays the number of points, normal and centroid."""
        s = "Points: \n"
        s += str(self.pts) + "\n"
        s += "Center: \n" + str(self.center) + "\n"
        s += "Normal: \n" + str(self.normal)
        return s

    def __eq__(self, other) -> bool:
        """Equality is defined as two fractures having the same index.

        Args:
            other: Fracture to be compared to self.

        Note:
            There are issues with this, behavior may change in the future.

        """
        return self.index == other.index

    def set_index(self, i: int):
        """Set index of this fracture.

        TODO: this is obsolete since the attribute `index` can be accessed directly

        Args:
            i (int): Index.

        """
        self.index = i

    def points(self) -> Generator[np.ndarray, None, None]:
        """Generator over the vertices of the fracture.

        Yields:
            (np.ndarray, nd x 1): Fracture vertex.

        """
        for i in range(self.pts.shape[1]):
            yield self.pts[:, i].reshape((-1, 1))

    def segments(self) -> Generator[np.ndarray, None, None]:
        """Generator over the segments according to the currently applied order.

        Yields:
            (np.ndarray, nd x 2): Polygon segment.

        """
        sz = self.pts.shape[1]
        for i in range(sz):
            yield self.pts[:, np.array([i, i + 1]) % sz]

    def is_vertex(self, p, tol: float = 1e-4) -> Tuple[bool, Union[int, None]]:
        """Check whether a given point is a vertex of the fracture.

        Args:
            p (np.ndarray, nd x 1):  Point to check.
            tol (float): Tolerance of point accuracy. Defaults to 1e-4.

        Returns:
            Tuple[bool, idx]: The bool is True, if `p` is a vertex of this fracture,
                false otherwise. If `p` is a vertex of this fracture, `idx` is an integer,
                representing the of the identical vertex. Otherwise, `idx` is None.

        """
        p = p.reshape((-1, 1))
        ap = np.hstack((p, self.pts))
        # TODO check if sqrt(3) is too much for other fractures than planes
        up, _, ind = setmembership.unique_columns_tol(ap, tol=tol * np.sqrt(3))

        # If the unique-operation did not remove any points, it is not a vertex.
        if up.shape[1] == ap.shape[1]:
            return False, None
        else:
            occurrences = np.where(ind == ind[0])[0]
            return True, (occurrences[1] - 1)

    def add_points(
        self,
        p: np.ndarray,
        check_convexity: bool = True,
        tol: float = 1e-4,
        enforce_pt_tol: Optional[float] = None,
    ) -> bool:
        """Add points to the polygon with the implemented sorting enforced.

        Always run a test to check that the points are still planar.
        By default, a check of convexity is also performed, however, this can be
        turned off to speed up simulations (the test uses sympy, which turns
        out to be slow in many cases).

        Args:
            p (np.ndarray, nd x n): Points to be added.
            check_convexity (bool, optional): Verify that the polygon is convex.
                Defaults to true.
            tol (float , optional): Tolerance used to check if the point already exists.
                Defaults to 1e-4.
            enforce_pt_tol (float, optional): Defaults to None. If not None, enforces a
                maximal distance between points by removing points failing that criterion.

        Returns:
            bool : True, if the resulting polygon is planar, False otherwise. If the
                argument `check_convexity` is True, checks for planarity *AND* convexity

        """
        to_enforce = np.hstack(
            (
                np.zeros(self.pts.shape[1], dtype=bool),
                np.ones(p.shape[1], dtype=bool),
            )
        )
        self.pts = np.hstack((self.pts, p))
        self.pts, _, _ = setmembership.unique_columns_tol(self.pts, tol=tol)

        # Sort points to counter-clockwise
        mask = self.sort_points()

        if enforce_pt_tol is not None:
            to_enforce = np.where(to_enforce[mask])[0]
            dist = pp.distances.pointset(self.pts)
            dist /= np.amax(dist)
            np.fill_diagonal(dist, np.inf)
            mask = np.where(dist < enforce_pt_tol)[0]
            mask = np.setdiff1d(mask, to_enforce, assume_unique=True)
            self.remove_points(mask)

        if check_convexity:
            return self.is_convex() and self.is_planar(tol)
        else:
            return self.is_planar()

    def remove_points(self, ind: Union[np.ndarray, List[int]], keep_orig: bool = False):
        """Remove points from the fracture definition.

        Args:
            ind (np.ndarray or list): Numpy array or list of indices of points to be removed.
            keep_orig (bool, optional): Defaults to False. If True, keeps the original
                points in the attribute :attr:`~Fracture.orig_pts`.

        """
        self.pts = np.delete(self.pts, ind, axis=1)

        if not keep_orig:
            self.orig_pts = self.pts

    def copy(self: FractureType) -> FractureType:
        """Return a copy of the fracture with the current vertices.

        Note:
            The original `points` (as given when the fracture was initialized) will NOT be
            preserved.

        Returns:
            FractureType: Fracture with the same points.

        """
        # deep copy points since mutable
        p = np.copy(self.pts)
        # TODO: Consider adding optional index parameter to this method.
        return type(self)(p, index=self.index)

    # Below are the dimension-dependent abstract methods
    @abc.abstractmethod
    def sort_points(self) -> np.ndarray:
        """Abstract method to sort the vertices as needed for geometric algorithms.

        Returns:
            np.ndarray(dtype=int): Indices corresponding to the sorting.

        """
        pass

    @abc.abstractmethod
    def local_coordinates(self) -> np.ndarray:
        """Abstract method for computing the local coordinates.

        The compuation is performed on the vertex coordinates in a local system and its
        local dimension `d`.

        For now, `d` is expected to be `nd` - 1, i.e. the fracture to be of co-dimension 1.

        Returns:
            np.ndarray((nd - 1) x npt) : Coordinates of the vertices in local dimensions.

        """
        pass

    @abc.abstractmethod
    def is_convex(self) -> bool:
        """Abstract method for implementing a convexity check.

        Returns:
            bool: True if the fracture is convex, False otherwise.

        """
        pass

    @abc.abstractmethod
    def is_planar(self, tol: float = 1e-4) -> bool:
        """Abstract method to check if the fracture is planar.

        Args:
            tol (float): Tolerance for non-planarity. Treated as an absolute quantity
                (no scaling with fracture extent).

        Returns:
            bool: True if the fracture is planar, False otherwise.

        """
        pass

    @abc.abstractmethod
    def compute_centroid(self) -> np.ndarray:
        """Abstract method for computing and returning the centroid of the fracture.

        Note that Convexity is assumed.

        """
        pass

    @abc.abstractmethod
    def compute_normal(self) -> np.ndarray:
        """Abstract method for computing and returning the normal vector."""
        pass

    @abc.abstractmethod
    def _check_pts(self) -> None:
        """Abstract method for checking consistency of `self.pts`.

        Raises:
            ValueError if self.pts violates some assumptions (e.g. shape).

        """
        pass
