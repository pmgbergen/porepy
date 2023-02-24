"""A module containing the abstract base class for all fractures."""

from __future__ import annotations

import abc
from typing import Generator, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from porepy.utils import setmembership


class Fracture(abc.ABC):
    """Abstract base class for representing a single fracture.

    This base class provides general functionalities agnostic to the dimension of the
    fracture and the ambient dimension. It contains various utility methods, mainly
    intended for the use together with the FractureNetwork class.

    A fracture is defined by its `num_points` vertices, stored in an `nd x num_points`
    numpy-array, where `nd` is the assumed ambient dimension. The order/sorting of
    vertices has to be implemented explicitly.

    Dimension-dependent routines are implemented as abstract methods.

    PorePy currently only supports planar and convex fractures fully. As a work-around,
    the fracture can be split into convex parts.

    Parameters:
        points: ``(shape=(nd, num_points))``
            Array containing the start- and end point/the corner
            points for line/plane fractures.
        tags: ``(shape=(num_tags)`, dtype=np.int8)``
            All the tags of the fracture. A tag value of ``-1`` equals to the tag not
            existing at all. Default is None.
        index: Identify the fracture with an index. Two fractures with the same index
            are assumed to be identical. Default is None.
        sort_points: Sort the points internally. Concrete implementation depends on the
            subclass. Default is True.

    """

    def __init__(
        self,
        points: ArrayLike,
        tags: Optional[ArrayLike] = None,
        index: Optional[int] = None,
        sort_points: bool = True,
    ):
        self.pts: np.ndarray = np.asarray(points, dtype=float)
        """Fracture vertices (shape=(nd, num_points)), stored in the implemented order.

        Note that the ``points`` passed to init will mutate.

        """
        self._check_pts()

        # Ensure that the points are sorted
        if sort_points:
            self.sort_points()

        self.normal: np.ndarray = self.compute_normal()
        """Normal vector `(shape=(nd, ))`."""
        self.center: np.ndarray = self.compute_centroid()
        """Centroid of the fracture `(shape=(nd, ))`."""
        self.orig_pts: np.ndarray = self.pts.copy()
        """Original fracture vertices `(shape=(nd, num_points))`.

        The original points are kept in case the fracture geometry is modified.

        """
        if tags is None:
            self.tags: np.ndarray = np.full((0,), -1, dtype=np.int8)
        else:
            self.tags = np.asarray(tags, dtype=np.int8)
        """Tags of the fracture.

        In the standard form, the first tag identifies the type of the fracture,
        referring to the numbering system in GmshInterfaceTags. The second tag keeps
        track of the numbering of the fracture (referring to the original order of the
        fractures) in geometry processing like intersection removal. Additional tags can
        be assigned by the user.

        A tag value of -1 means that the fracture does not have the specified tag. This
        enables e.g. the functionality for a fracture to have the second tag, but no the
        first one. In a more extreme case, ``fracture.tags==[-1, -1, -1, -1, -1]`` is
        equal to the fracture not having any tags at all.

        """

        self.index: Optional[int] = index
        """Index of fracture.

        Intended use in :class:`~porepy.fracs.fracture_network_2d.FractureNetwork2d`
        and :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`. Exact use is
        not clear (several fractures can be given same index), use with care.

        """

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

    def __eq__(self, other: object) -> bool:
        """Equality is defined as two fractures having the same index.

        Parameters:
            other: Fracture to be compared to self.

        Returns:
            True if the fractures have the same index, False otherwise.

        Note:
            There are issues with this, behavior may change in the future.

        """
        if not isinstance(other, Fracture):
            return NotImplemented
        return self.index == other.index

    def set_index(self, i: int):
        """Set index of this fracture.

        Parameters:
            i: Index.

        """
        self.index = i

    def points(self) -> Generator[np.ndarray, None, None]:
        """Generator over the vertices of the fracture.

        Yields:
            Fracture vertex `(shape=(nd, 1))`.

        """
        for i in range(self.pts.shape[1]):
            yield self.pts[:, i].reshape((-1, 1))

    def segments(self) -> Generator[np.ndarray, None, None]:
        """Generator over the segments according to the currently applied order.

        Yields:
            Fracture segment `(shape=(nd, 2))`.

        """
        sz = self.pts.shape[1]
        for i in range(sz):
            yield self.pts[:, np.array([i, i + 1]) % sz]

    def is_vertex(
        self, p: np.ndarray, tol: float = 1e-4
    ) -> tuple[bool, Union[int, None]]:
        """Check whether a given point is a vertex of the fracture.

        Parameters:
            p (shape=(nd, )): Point to be checked.
            tol: Tolerance of point accuracy. Default is ``1e-4``.

        Returns:
            A tuple containing

            bool:
                Indicates whether the point is a vertex.
            ndarray:
                Gives the position of ``p`` in :attr:`pts` if the point is a vertex.
                Else, None is returned.

        """
        p = p.reshape((-1, 1))
        ap = np.hstack((p, self.pts))
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
            The original ``points`` (as given when the fracture was initialized) will
            *not* be preserved.

        Returns:
            Fracture with the same points.

        """
        return type(self)(self.pts.copy(), index=self.index)

    # Below are the dimension-dependent abstract methods
    @abc.abstractmethod
    def sort_points(self) -> np.ndarray:
        """Abstract method to sort the vertices as needed for geometric algorithms.

        Returns:
            Array of integer indices corresponding to the sorting.

        """
        pass

    @abc.abstractmethod
    def local_coordinates(self) -> np.ndarray:
        """Abstract method for computing the local coordinates.

        The compuation is performed on the vertex coordinates in a local system and its
        local dimension :math:`d` is assumed to be :math:`d = nd - 1`, i.e. the fracture
        has co-dimension 1.

        Returns:
            Coordinates of the vertices in local dimensions `(shape=(d, num_points))`.

        """
        pass

    @abc.abstractmethod
    def compute_centroid(self) -> np.ndarray:
        """Abstract method for computing and returning the centroid of the fracture.

        Note that convexity is assumed.

        """
        pass

    @abc.abstractmethod
    def compute_normal(self) -> np.ndarray:
        """Abstract method for computing and returning the normal vector."""
        pass

    @abc.abstractmethod
    def _check_pts(self) -> None:
        """Abstract method for checking consistency of :attr:`pts`.

        Raises:
            ValueError if :attr:`pts` violates some assumptions (e.g. shape).

        """
        pass
