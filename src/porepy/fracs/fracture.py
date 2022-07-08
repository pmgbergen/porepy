"""A module containing the abstract base class for all fractures."""

from __future__ import annotations

import abc
from typing import Generator, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from porepy.utils import setmembership


class Fracture(abc.ABC):
    """Abstract base class for representing a single fracture.

    This base class provides general functionalities agnostic to the dimension of
    the fracture and the ambient dimension. It contains various utility methods,
    mainly intended for the use together with the FractureNetwork class.

    A fracture is defined by its `npt` vertices, stored in an `nd x npt` numpy-array,
    where `nd` is the assumed ambient dimension. The order/sorting of vertices has to be
    implemented explicitly.

    Dimension-dependent routines are implemented as abstract methods.

    PorePy currently only supports planar and convex fractures fully. As a work-around,
    the fracture can be split into convex parts.

    Attributes:
        points (ArrayLike): Fracture vertices, stored in the implemented order.
            Shape is (nd, npt).
        orig_pts (np.ndarray): Original fracture vertices.
            Shape is (nd, npt)
            The original points are kept in case the fracture geometry is modified.
        center (np.ndarray): Fracture centroid.
            Shape is (nd, ).
        normal (np.ndarray): Normal vector.
            Shape is (nd, ).
        index (int): Index of fracture.
            Intended use in FractureNetwork. Exact use is not clear (several fractures can
            be given same index), use it with care.

    """

    def __init__(
        self,
        points: ArrayLike,
        index: Optional[int] = None,
        sort_points: bool = True,
    ):
        """Instatiation of the abstract fracture class.

        The constructor sets the vertices of the fracture and computes the centroid and
        normals. Optionally, checks for planarity and convexity are performed.

        Args:
            points (ArrayLike): Vertices of new fracture.
                Shape is (nd, npt).
                How many vertices are necessary depends on the dimension. Note that `points`
                can be an np.ndarray or any suitable object that can be converted into an
                np.ndarray via np.asarray(`points`).
            index (int, optional): Index of fracture. Default is None.
            sort_points (bool , optional): If True, sorts the points according to the
                implemented order. Defaults to True.
                Convexity may play a role, thus if a fracture is known to be non-convex,
                the parameter should be False.

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
            (np.ndarray, nd x 2): Fracture segment.

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

    def copy(self) -> Fracture:
        """Return a copy of the fracture with the current vertices.

        Note:
            The original `points` (as given when the fracture was initialized) will NOT be
            preserved.

        Returns:
            FractureType: Fracture with the same points.

        """
        return type(self)(self.pts.copy(), index=self.index)

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
